# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
import pytest
import torch
import torch.nn.functional as F

from chunk_gated_delta_rule import chunk_gated_delta_rule

pytestmark = pytest.mark.skip(
    reason="chunk_gated_delta_rule is superseded by the fused mega_gdn ops and no "
    "longer used by models; solve_tril_16x16_kernel also hits a bishengir-compile-a5 "
    "SubViewOp assertion on Ascend950PR"
)

CHUNK_SIZE = 64


def _has_npu() -> bool:
    try:
        _ = torch.zeros((1,), device="npu:0")
        return True
    except Exception:
        return False


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.square().sum(dim=dim, keepdim=True) + eps)


def _expand_qk_to_v_heads(x: torch.Tensor, num_v_heads: int) -> torch.Tensor:
    h_qk = x.shape[1]
    if h_qk == num_v_heads:
        return x
    if num_v_heads % h_qk != 0:
        raise ValueError(f"Invalid grouped heads: Hqk={h_qk}, Hv={num_v_heads}.")
    group_size = num_v_heads // h_qk
    return x.repeat_interleave(group_size, dim=1)


def _torch_chunk_gated_delta_rule_chunked(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)

    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))

    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

    mask_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=0,
    )

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_diag, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=value.dtype)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)

    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device),
        diagonal=1,
    )

    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_inter_chunk = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_upper, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        inter_state = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = inter_state + attn_inter_chunk @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0],
        core_attn_out.shape[1],
        -1,
        core_attn_out.shape[-1],
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if head_first:
        raise DeprecationWarning("head_first=True is not supported in the reference path.")
    if cu_seqlens is None:
        raise ValueError("Reference path expects varlen inputs with cu_seqlens.")
    if q.shape[0] != 1:
        raise ValueError("Variable-length mode expects batch size B=1.")

    batch_size, total_tokens, h_qk, k_dim = q.shape
    h_v = v.shape[2]
    v_dim = v.shape[-1]
    if k.shape != q.shape:
        raise ValueError("q and k shapes must match.")
    if g.shape != beta.shape or g.shape[:2] != (batch_size, total_tokens) or g.shape[2] != h_v:
        raise ValueError("g/beta must have shape [B, T, Hv] matching v.")

    states = (
        initial_state.to(torch.float32).clone()
        if initial_state is not None
        else torch.zeros(len(cu_seqlens) - 1, h_v, k_dim, v_dim, dtype=torch.float32)
    )
    out = torch.zeros_like(v)

    for seq_idx in range(len(cu_seqlens) - 1):
        start = int(cu_seqlens[seq_idx].item())
        end = int(cu_seqlens[seq_idx + 1].item())
        if end <= start:
            continue

        q_seq = _expand_qk_to_v_heads(q[0, start:end], h_v).unsqueeze(0)
        k_seq = _expand_qk_to_v_heads(k[0, start:end], h_v).unsqueeze(0)
        v_seq = v[0, start:end].unsqueeze(0)
        g_seq = g[0, start:end].unsqueeze(0)
        beta_seq = beta[0, start:end].unsqueeze(0)
        init_seq_state = states[seq_idx].unsqueeze(0)

        out_seq, final_state = _torch_chunk_gated_delta_rule_chunked(
            query=q_seq,
            key=k_seq,
            value=v_seq,
            g=g_seq,
            beta=beta_seq,
            chunk_size=CHUNK_SIZE,
            initial_state=init_seq_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
        out[0, start:end] = out_seq[0]
        states[seq_idx] = final_state[0]

    if output_final_state:
        return out, states
    return out, None


@pytest.mark.skipif(not _has_npu(), reason="NPU device not available")
@pytest.mark.parametrize(
    "seed,q_shape,v_shape,initial_state_shape,cu_seqlens,use_qk_l2norm_in_kernel",
    [
        (
            1234,
            (1, 31, 4, 128),
            (1, 31, 8, 128),
            (1, 8, 128, 128),
            [0, 31],
            True,
        ),
    ],
)
def test_chunk_gated_delta_rule_varlen_against_ref(
    seed,
    q_shape,
    v_shape,
    initial_state_shape,
    cu_seqlens,
    use_qk_l2norm_in_kernel,
):
    torch.manual_seed(seed)
    cu_seqlens_cpu = torch.tensor(cu_seqlens, dtype=torch.int32)

    q_cpu = torch.randn(*q_shape, dtype=torch.bfloat16)
    k_cpu = torch.randn(*q_shape, dtype=torch.bfloat16)
    v_cpu = torch.randn(*v_shape, dtype=torch.bfloat16)
    g_cpu = F.logsigmoid(torch.randn(*v_shape[:-1], dtype=torch.float32))
    beta_cpu = torch.sigmoid(torch.randn(*v_shape[:-1], dtype=torch.float32)).to(torch.bfloat16)
    initial_state_cpu = torch.randn(*initial_state_shape, dtype=torch.bfloat16)

    ref_out, ref_final_state = chunk_gated_delta_rule_ref(
        q=q_cpu,
        k=k_cpu,
        v=v_cpu,
        g=g_cpu,
        beta=beta_cpu,
        initial_state=initial_state_cpu,
        output_final_state=True,
        cu_seqlens=cu_seqlens_cpu,
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )

    out_npu, final_state_npu = chunk_gated_delta_rule(
        q=q_cpu.npu(),
        k=k_cpu.npu(),
        v=v_cpu.npu(),
        g=g_cpu.npu(),
        beta=beta_cpu.npu(),
        initial_state=initial_state_cpu.npu(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_cpu.npu(),
        head_first=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    out_npu = out_npu.cpu()
    final_state_npu = final_state_npu.cpu()

    torch.testing.assert_close(ref_out.float(), out_npu.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(
        ref_final_state.float(),
        final_state_npu.float(),
        atol=1e-2,
        rtol=1e-2,
    )
