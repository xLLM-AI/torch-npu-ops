/* Copyright 2026 The xLLM Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://github.com/jd-opensource/xllm/blob/main/LICENSE
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==============================================================================
 */

#include "operation_factory.h"
#include "triton_ops_api.h"

#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>

#include <vector>

namespace xllm::kernel::npu {

namespace {

bool is_supported_initial_state_dtype(torch::ScalarType dtype) {
  return dtype == torch::kBFloat16 || dtype == torch::kFloat32;
}

int64_t cdiv(int64_t a, int64_t b) {
  TORCH_CHECK(b > 0, "cdiv divisor must be positive, got ", b);
  return (a + b - 1) / b;
}


std::optional<torch::Tensor> prepare_chunk_indices_like_python(
    const std::optional<torch::Tensor>& cu_seqlens,
    int64_t chunk_size) {
  if (!cu_seqlens.has_value()) {
    return std::nullopt;
  }

  auto cu_cpu = cu_seqlens.value().to(torch::kCPU).contiguous();
  std::vector<int64_t> indices_host;
  const int64_t num_sequences = cu_cpu.numel() - 1;
  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    int64_t start = 0;
    int64_t end = 0;
    if (cu_cpu.scalar_type() == torch::kInt32) {
      auto* ptr = cu_cpu.data_ptr<int32_t>();
      start = static_cast<int64_t>(ptr[seq_idx]);
      end = static_cast<int64_t>(ptr[seq_idx + 1]);
    } else {
      auto* ptr = cu_cpu.data_ptr<int64_t>();
      start = ptr[seq_idx];
      end = ptr[seq_idx + 1];
    }

    const int64_t num_chunks = cdiv(end - start, chunk_size);
    for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      indices_host.push_back(seq_idx);
      indices_host.push_back(chunk_idx);
    }
  }

  auto options = torch::TensorOptions()
                     .dtype(cu_seqlens.value().scalar_type())
                     .device(cu_seqlens.value().device());
  if (indices_host.empty()) {
    return torch::empty({0, 2}, options);
  }
  return torch::tensor(indices_host, options).reshape({-1, 2}).contiguous();
}

std::optional<torch::Tensor> prepare_chunk_offsets_like_python(
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& chunk_offsets,
    int64_t chunk_size) {
  if (chunk_offsets.has_value()) {
    return chunk_offsets.value().to(torch::kInt64).contiguous();
  }
  if (!cu_seqlens.has_value()) {
    return std::nullopt;
  }

  auto cu_cpu = cu_seqlens.value().to(torch::kCPU).contiguous();
  std::vector<int64_t> offsets_host;
  offsets_host.reserve(cu_cpu.numel());
  offsets_host.push_back(0);

  if (cu_cpu.scalar_type() == torch::kInt32) {
    auto* cu_ptr = cu_cpu.data_ptr<int32_t>();
    for (int64_t i = 0; i < cu_cpu.numel() - 1; ++i) {
      int64_t len =
          static_cast<int64_t>(cu_ptr[i + 1]) - static_cast<int64_t>(cu_ptr[i]);
      offsets_host.push_back(offsets_host.back() + cdiv(len, chunk_size));
    }
  } else {
    auto* cu_ptr = cu_cpu.data_ptr<int64_t>();
    for (int64_t i = 0; i < cu_cpu.numel() - 1; ++i) {
      int64_t len = cu_ptr[i + 1] - cu_ptr[i];
      offsets_host.push_back(offsets_host.back() + cdiv(len, chunk_size));
    }
  }

  return torch::tensor(
             offsets_host,
             torch::TensorOptions()
                 .dtype(torch::kInt64)
                 .device(cu_seqlens.value().device()))
      .contiguous();
}

int64_t infer_total_chunks(
    int64_t B,
    int64_t T,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& chunk_offsets) {
  if (!cu_seqlens.has_value()) {
    return cdiv(T, chunk_size);
  }
  if (!chunk_offsets.has_value()) {
    return cdiv(T, chunk_size);
  }

  auto chunk_offsets_cpu = chunk_offsets.value().to(torch::kCPU).contiguous();
  if (chunk_offsets_cpu.numel() == 0) {
    return 0;
  }
  if (chunk_offsets_cpu.scalar_type() == torch::kInt32) {
    return static_cast<int64_t>(
        chunk_offsets_cpu.data_ptr<int32_t>()[chunk_offsets_cpu.numel() - 1]);
  }
  return chunk_offsets_cpu.data_ptr<int64_t>()[chunk_offsets_cpu.numel() - 1];
}

void record_tensor_if_needed(const torch::Tensor& tensor,
                             c10_npu::NPUStream stream) {
  if (!tensor.defined() || !tensor.device().is_privateuseone()) {
    return;
  }
  c10_npu::NPUCachingAllocator::recordStream(tensor.storage().data_ptr(),
                                             stream);
}

}  // namespace

torch::Tensor npu_chunk_local_cumsum(
    const torch::Tensor& g2,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens) {
  torch::Tensor g = g2.to(torch::kFloat32);
  TORCH_CHECK(g.dim() == 3, "chunk_local_cumsum expects g to have shape [B, T, H].");
  TORCH_CHECK(g.scalar_type() == torch::kFloat32,
              "chunk_local_cumsum expects g to be float32 to match the "
              "compiled Triton ABI, got ",
              g.scalar_type());
  TORCH_CHECK(chunk_size > 0 && (chunk_size & (chunk_size - 1)) == 0,
              "chunk_local_cumsum expects chunk_size to be a power of two, got ",
              chunk_size);

  const int64_t B = g.size(0);
  const int64_t T = g.size(1);
  const int64_t H = g.size(2);
  // if (cu_seqlens.has_value()) {
  //   TORCH_CHECK(B == 1,
  //               "chunk_local_cumsum only supports B=1 when cu_seqlens is "
  //               "provided.");
  // }
  TORCH_CHECK(H <= 64,
              "chunk_local_cumsum runtime-variable adapter currently supports "
              "H <= 64, got ",
              H);

  const int64_t block_t = 512;
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  auto block_indices = prepare_chunk_indices_like_python(cu_prepared, block_t);
  if (block_indices.has_value()) {
    block_indices = block_indices.value().to(torch::kInt32).contiguous();
  }
  const int64_t num_blocks =
      cu_prepared.has_value() ? block_indices.value().size(0) : cdiv(T, block_t);

  auto g_contig = g.contiguous();
  auto out = torch::empty(
      g_contig.sizes(),
      torch::TensorOptions().dtype(torch::kFloat32).device(g.device()));

  auto npu_stream = c10_npu::getCurrentNPUStream(g_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());
  void* cu_ptr = cu_prepared.has_value() ? cu_prepared.value().data_ptr() : nullptr;
  void* block_indices_ptr =
      block_indices.has_value() ? block_indices.value().data_ptr() : nullptr;

  auto& op = OperationFactory::instance().chunk_local_cumsum_scalar();
  auto ret = op.execute(stream,
                        static_cast<int32_t>(num_blocks),
                        static_cast<int32_t>(B),
                        1,
                        [&](ArgsBuilder& ab) {
                          ab.constructArgs(g_contig.data_ptr(),
                                           out.data_ptr(),
                                           cu_ptr,
                                           block_indices_ptr,
                                           static_cast<int32_t>(T),
                                           static_cast<int32_t>(H));
                        });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'chunk_local_cumsum_scalar_kernel': "
               << ret;
  }

  record_tensor_if_needed(g_contig, npu_stream);
  if (cu_prepared.has_value()) {
    record_tensor_if_needed(cu_prepared.value(), npu_stream);
  }
  if (block_indices.has_value()) {
    record_tensor_if_needed(block_indices.value(), npu_stream);
  }
  return out;
}

torch::Tensor npu_chunk_scaled_dot_kkt_fwd(
    const torch::Tensor& k,
    const torch::Tensor& beta,
    const torch::Tensor& g_cumsum,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens) {
  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t H = beta.size(2);

  auto k_contig = k.contiguous();
  auto beta_prepared = beta.transpose(1, 2).contiguous();
  auto g_prepared = g_cumsum.transpose(1, 2).contiguous();
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  auto chunk_indices = prepare_chunk_indices_like_python(cu_prepared, chunk_size);
  if (chunk_indices.has_value()) {
    chunk_indices = chunk_indices.value().to(torch::kInt32).contiguous();
  }

  const int64_t nt =
      cu_prepared.has_value() ? chunk_indices.value().size(0) : cdiv(T, chunk_size);
  auto A = torch::empty({B, T, H, chunk_size},
                        torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));

  auto npu_stream = c10_npu::getCurrentNPUStream(k_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());
  void* cu_ptr = cu_prepared.has_value() ? cu_prepared.value().data_ptr() : nullptr;
  void* chunk_indices_ptr =
      chunk_indices.has_value() ? chunk_indices.value().data_ptr() : nullptr;

  auto& op = OperationFactory::instance().chunk_scaled_dot_kkt_fwd();
  auto ret = op.execute(stream,
                        static_cast<int32_t>(nt),
                        static_cast<int32_t>(B * H),
                        1,
                        [&](ArgsBuilder& ab) {
                          ab.constructArgs(k_contig.data_ptr(),
                                           beta_prepared.data_ptr(),
                                           g_prepared.data_ptr(),
                                           A.data_ptr(),
                                           cu_ptr,
                                           chunk_indices_ptr,
                                           static_cast<int32_t>(T),
                                           static_cast<int32_t>(B),
                                           static_cast<int32_t>(H),
                                           static_cast<int32_t>(Hg),
                                           static_cast<int32_t>(K));
                        });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'chunk_scaled_dot_kkt_fwd_kernel': "
               << ret;
  }

  record_tensor_if_needed(k_contig, npu_stream);
  record_tensor_if_needed(beta_prepared, npu_stream);
  record_tensor_if_needed(g_prepared, npu_stream);
  if (cu_prepared.has_value()) {
    record_tensor_if_needed(cu_prepared.value(), npu_stream);
  }
  if (chunk_indices.has_value()) {
    record_tensor_if_needed(chunk_indices.value(), npu_stream);
  }
  return A;
}

torch::Tensor npu_solve_tril(
    const torch::Tensor& A,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens,
    torch::ScalarType output_dtype) {
  TORCH_CHECK(chunk_size == 64,
              "chunk_gated_delta_rule solve_tril wrapper only supports "
              "chunk_size=64.");
  const int64_t B = A.size(0);
  const int64_t T = A.size(1);
  const int64_t H = A.size(2);
  const int64_t solve_block_t = 16;

  auto A_contig = A.contiguous();
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  auto solve_block_indices =
      prepare_chunk_indices_like_python(cu_prepared, solve_block_t);
  if (solve_block_indices.has_value()) {
    solve_block_indices =
        solve_block_indices.value().to(torch::kInt32).contiguous();
  }
  const int64_t solve_nt = cu_prepared.has_value()
                               ? solve_block_indices.value().size(0)
                               : cdiv(T, solve_block_t);
  auto Ad = torch::empty({B, T, H, 16},
                         torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));

  auto npu_stream = c10_npu::getCurrentNPUStream(A_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());
  void* cu_ptr = cu_prepared.has_value() ? cu_prepared.value().data_ptr() : nullptr;
  void* solve_indices_ptr = solve_block_indices.has_value()
                                ? solve_block_indices.value().data_ptr()
                                : nullptr;

  auto& solve_op = OperationFactory::instance().solve_tril_16x16();
  auto ret = solve_op.execute(stream,
                              static_cast<int32_t>(solve_nt),
                              static_cast<int32_t>(B * H),
                              1,
                              [&](ArgsBuilder& ab) {
                                ab.constructArgs(A_contig.data_ptr(),
                                                 Ad.data_ptr(),
                                                 cu_ptr,
                                                 solve_indices_ptr,
                                                 static_cast<int32_t>(T),
                                                 static_cast<int32_t>(H));
                              });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'solve_tril_16x16_kernel': " << ret;
  }

  auto merge_indices = prepare_chunk_indices_like_python(cu_prepared, chunk_size);
  if (merge_indices.has_value()) {
    merge_indices = merge_indices.value().to(torch::kInt32).contiguous();
  }
  const int64_t nt =
      cu_prepared.has_value() ? merge_indices.value().size(0) : cdiv(T, chunk_size);
  auto Ai = torch::empty({B, T, H, chunk_size},
                         torch::TensorOptions().dtype(output_dtype).device(A.device()));
  void* merge_indices_ptr =
      merge_indices.has_value() ? merge_indices.value().data_ptr() : nullptr;

  auto& merge_op = OperationFactory::instance().merge_16x16_to_64x64_inverse();
  ret = merge_op.execute(stream,
                         static_cast<int32_t>(nt),
                         static_cast<int32_t>(B * H),
                         1,
                         [&](ArgsBuilder& ab) {
                           ab.constructArgs(A_contig.data_ptr(),
                                            Ad.data_ptr(),
                                            Ai.data_ptr(),
                                            cu_ptr,
                                            merge_indices_ptr,
                                            static_cast<int32_t>(T),
                                            static_cast<int32_t>(H));
                         });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'merge_16x16_to_64x64_inverse_kernel': "
               << ret;
  }

  record_tensor_if_needed(A_contig, npu_stream);
  if (cu_prepared.has_value()) {
    record_tensor_if_needed(cu_prepared.value(), npu_stream);
  }
  if (solve_block_indices.has_value()) {
    record_tensor_if_needed(solve_block_indices.value(), npu_stream);
  }
  if (merge_indices.has_value()) {
    record_tensor_if_needed(merge_indices.value(), npu_stream);
  }
  return Ai;
}

std::pair<torch::Tensor, torch::Tensor> npu_recompute_w_u_fwd(
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& beta,
    const torch::Tensor& g_cumsum,
    const torch::Tensor& A,
    const std::optional<torch::Tensor>& cu_seqlens) {
  const int64_t B = k.size(0);
  const int64_t T = k.size(1);
  const int64_t Hg = k.size(2);
  const int64_t K = k.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);
  const int64_t chunk_size = A.size(3);

  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  auto beta_prepared = beta.transpose(1, 2).contiguous();
  auto g_prepared = g_cumsum.transpose(1, 2).contiguous();
  auto A_contig = A.contiguous();
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  auto chunk_indices = prepare_chunk_indices_like_python(cu_prepared, chunk_size);
  if (chunk_indices.has_value()) {
    chunk_indices = chunk_indices.value().to(torch::kInt32).contiguous();
  }

  const int64_t nt =
      cu_prepared.has_value() ? chunk_indices.value().size(0) : cdiv(T, chunk_size);
  auto w = torch::empty({B, T, H, k.size(3)},
                        torch::TensorOptions().dtype(k.dtype()).device(k.device()));
  auto u = torch::empty_like(v_contig);

  auto npu_stream = c10_npu::getCurrentNPUStream(k_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());
  void* cu_ptr = cu_prepared.has_value() ? cu_prepared.value().data_ptr() : nullptr;
  void* chunk_indices_ptr =
      chunk_indices.has_value() ? chunk_indices.value().data_ptr() : nullptr;

  auto& op = OperationFactory::instance().recompute_w_u_fwd();
  auto ret = op.execute(stream,
                        static_cast<int32_t>(nt),
                        static_cast<int32_t>(B * H),
                        1,
                        [&](ArgsBuilder& ab) {
                          ab.constructArgs(k_contig.data_ptr(),
                                           v_contig.data_ptr(),
                                           beta_prepared.data_ptr(),
                                           w.data_ptr(),
                                           u.data_ptr(),
                                           A_contig.data_ptr(),
                                           g_prepared.data_ptr(),
                                           cu_ptr,
                                           chunk_indices_ptr,
                                           static_cast<int32_t>(T),
                                           static_cast<int32_t>(H),
                                           static_cast<int32_t>(Hg),
                                           static_cast<int32_t>(K),
                                           static_cast<int32_t>(V));
                        });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'recompute_w_u_fwd_kernel': " << ret;
  }

  record_tensor_if_needed(k_contig, npu_stream);
  record_tensor_if_needed(v_contig, npu_stream);
  record_tensor_if_needed(beta_prepared, npu_stream);
  record_tensor_if_needed(g_prepared, npu_stream);
  record_tensor_if_needed(A_contig, npu_stream);
  if (cu_prepared.has_value()) {
    record_tensor_if_needed(cu_prepared.value(), npu_stream);
  }
  if (chunk_indices.has_value()) {
    record_tensor_if_needed(chunk_indices.value(), npu_stream);
  }
  return {w, u};
}

torch::Tensor npu_chunk_fwd_o(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& h,
    const torch::Tensor& g_cumsum,
    float scale,
    int64_t chunk_size,
    const std::optional<torch::Tensor>& cu_seqlens) {
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hg = q.size(2);
  const int64_t K = q.size(3);
  const int64_t H = v.size(2);
  const int64_t V = v.size(3);
  const int64_t bv = 128;

  auto q_contig = q.contiguous();
  auto k_contig = k.contiguous();
  auto v_contig = v.contiguous();
  auto h_contig = h.contiguous();
  auto g_prepared = g_cumsum.transpose(1, 2).contiguous();
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  // chunk_fwd_kernel_o indexes varlen work by a dense [sequence, chunk] table.
  // Keep this metadata in the same int32 ABI form as cu_seqlens; the older
  // prefix-offset vector is not interchangeable and can make the kernel read
  // past the metadata buffer on A5.
  auto chunk_indices = prepare_chunk_indices_like_python(cu_prepared, chunk_size);
  if (chunk_indices.has_value()) {
    chunk_indices = chunk_indices.value().to(torch::kInt32).contiguous();
  }
  const int64_t total_chunks =
      cu_prepared.has_value() ? chunk_indices.value().size(0)
                              : B * cdiv(T, chunk_size);
  auto out = torch::empty_like(v_contig);

  auto npu_stream = c10_npu::getCurrentNPUStream(q_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());
  void* cu_ptr = cu_prepared.has_value() ? cu_prepared.value().data_ptr() : nullptr;
  void* chunk_indices_ptr =
      chunk_indices.has_value() ? chunk_indices.value().data_ptr() : nullptr;

  auto& op = OperationFactory::instance().chunk_fwd_o();
  auto ret = op.execute(stream,
                        static_cast<int32_t>(cdiv(V, bv)),
                        static_cast<int32_t>(total_chunks),
                        static_cast<int32_t>(H),
                        [&](ArgsBuilder& ab) {
                          ab.constructArgs(q_contig.data_ptr(),
                                           k_contig.data_ptr(),
                                           v_contig.data_ptr(),
                                           h_contig.data_ptr(),
                                           g_prepared.data_ptr(),
                                           out.data_ptr(),
                                           cu_ptr,
                                           chunk_indices_ptr,
                                           scale,
                                           static_cast<int32_t>(T),
                                           static_cast<int32_t>(H),
                                           static_cast<int32_t>(Hg),
                                           static_cast<int32_t>(K),
                                           static_cast<int32_t>(V));
                        });
  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for 'chunk_fwd_kernel_o': " << ret;
  }

  record_tensor_if_needed(q_contig, npu_stream);
  record_tensor_if_needed(k_contig, npu_stream);
  record_tensor_if_needed(v_contig, npu_stream);
  record_tensor_if_needed(h_contig, npu_stream);
  record_tensor_if_needed(g_prepared, npu_stream);
  if (cu_prepared.has_value()) {
    record_tensor_if_needed(cu_prepared.value(), npu_stream);
  }
  if (chunk_indices.has_value()) {
    record_tensor_if_needed(chunk_indices.value(), npu_stream);
  }
  return out;
}

std::pair<torch::Tensor, torch::Tensor> npu_chunk_gated_delta_rule(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    const std::optional<float>& scale,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const std::optional<torch::Tensor>& cu_seqlens,
    bool head_first,
    bool use_qk_l2norm_in_kernel) {
  TORCH_CHECK(!head_first,
              "chunk_gated_delta_rule torch_api only supports head_first=false.");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
                  k.scalar_type() == torch::kBFloat16 &&
                  v.scalar_type() == torch::kBFloat16,
              "chunk_gated_delta_rule expects q/k/v to be bfloat16.");
  if (initial_state.has_value()) {
    TORCH_CHECK(is_supported_initial_state_dtype(initial_state.value().scalar_type()),
                "chunk_gated_delta_rule expects initial_state to be bfloat16 "
                "or float32, got ",
                initial_state.value().scalar_type());
  }

  const auto input_dtype = q.scalar_type();
  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t Hqk = q.size(2);
  const int64_t H = v.size(2);
  const int64_t K = q.size(3);
  const int64_t chunk_size = 64;
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have the same shape.");
  TORCH_CHECK(H % Hqk == 0,
              "chunk_gated_delta_rule expects num_heads_v to be divisible by "
              "num_heads_qk, got ",
              H,
              " and ",
              Hqk);
  TORCH_CHECK(beta.dim() == 3 && beta.size(0) == B && beta.size(1) == T &&
                  beta.size(2) == H,
              "beta must have shape [B, T, H].");
  TORCH_CHECK(g.dim() == 3 && g.size(0) == B && g.size(1) == T &&
                  g.size(2) == H,
              "g must have shape [B, T, H].");
  // if (cu_seqlens.has_value()) {
  //   TORCH_CHECK(B == 1,
  //               "chunk_gated_delta_rule only supports B=1 when cu_seqlens is "
  //               "provided, got B=",
  //               B);
  // }

  auto q_prepared = use_qk_l2norm_in_kernel ? npu_l2norm_last_dim(q) : q;
  auto k_prepared = use_qk_l2norm_in_kernel ? npu_l2norm_last_dim(k) : k;
  auto cu_prepared = cu_seqlens.has_value()
                         ? std::optional<torch::Tensor>(
                               cu_seqlens.value().to(torch::kInt32).contiguous())
                         : std::nullopt;
  auto g_cumsum = npu_chunk_local_cumsum(g, chunk_size, cu_prepared);
  const float scale_value =
      scale.has_value() ? scale.value() : std::pow(static_cast<float>(K), -0.5f);
  auto A = npu_chunk_scaled_dot_kkt_fwd(
      k_prepared, beta, g_cumsum, chunk_size, cu_prepared);
  auto A_inv = npu_solve_tril(A, chunk_size, cu_prepared, k.scalar_type());
  auto [w, u] =
      npu_recompute_w_u_fwd(k_prepared, v, beta, g_cumsum, A_inv, cu_prepared);
  auto init_state_prepared =
      initial_state.has_value()
          ? std::optional<torch::Tensor>(
                initial_state.value().to(torch::kFloat32).contiguous())
          : std::nullopt;
  auto [h, v_new, final_state] = npu_chunk_gated_delta_rule_fwd_h(
      k_prepared,
      w,
      u,
      g_cumsum,
      init_state_prepared,
      output_final_state,
      chunk_size,
      true,
      cu_prepared,
      std::nullopt);
  auto out = npu_chunk_fwd_o(
      q_prepared, k_prepared, v_new, h, g_cumsum, scale_value, chunk_size, cu_prepared);

  return {out.to(input_dtype), output_final_state ? final_state : torch::Tensor()};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> npu_chunk_gated_delta_rule_fwd_h(
    torch::Tensor& k,
    torch::Tensor& w,
    torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::Tensor>& chunk_offsets) {
  TORCH_CHECK(chunk_size == 64,
              "chunk_gated_delta_rule_fwd_h only supports chunk_size=64 in "
              "the current compiled binary.");
  TORCH_CHECK(k.scalar_type() == torch::kBFloat16,
              "chunk_gated_delta_rule_fwd_h expects k to be bfloat16, got ",
              k.scalar_type());
  TORCH_CHECK(w.scalar_type() == torch::kBFloat16,
              "chunk_gated_delta_rule_fwd_h expects w to be bfloat16, got ",
              w.scalar_type());
  TORCH_CHECK(u.scalar_type() == torch::kBFloat16,
              "chunk_gated_delta_rule_fwd_h expects u to be bfloat16, got ",
              u.scalar_type());
  if (initial_state.has_value()) {
    TORCH_CHECK(is_supported_initial_state_dtype(initial_state.value().scalar_type()),
                "chunk_gated_delta_rule_fwd_h expects initial_state to be "
                "bfloat16 or float32, got ",
                initial_state.value().scalar_type());
  }

  auto k_shape = k.sizes();
  auto u_shape = u.sizes();

  int64_t B = k_shape[0];
  int64_t T = k_shape[1];
  int64_t Hg = k_shape[2];
  int64_t K = k_shape[3];
  int64_t H = u_shape[2];
  int64_t V = u_shape[3];

  auto k_contig = k.contiguous();
  auto w_contig = w.contiguous();
  auto u_contig = u.contiguous();
  auto g_prepared =
      g.has_value()
          ? std::optional<torch::Tensor>(g.value()
                                             .to(torch::kFloat32)
                                             .transpose(1, 2)
                                             .contiguous())
          : std::nullopt;
  auto initial_state_prepared =
      initial_state.has_value()
          ? std::optional<torch::Tensor>(
                initial_state.value().to(torch::kFloat32).contiguous())
          : std::nullopt;
  auto cu_seqlens_prepared =
      cu_seqlens.has_value()
          ? std::optional<torch::Tensor>(
                cu_seqlens.value().to(torch::kInt32).contiguous())
          : std::nullopt;

  std::optional<torch::Tensor> chunk_offsets_prepared =
      prepare_chunk_offsets_like_python(
          cu_seqlens_prepared, chunk_offsets, chunk_size);

  int64_t N = B;
  if (cu_seqlens_prepared.has_value()) {
    N = cu_seqlens_prepared.value().numel() - 1;
  }
  int64_t NT = infer_total_chunks(
      B, T, chunk_size, cu_seqlens_prepared, chunk_offsets_prepared);

  torch::Tensor h =
      torch::empty({B, NT, H, K, V},
                   torch::TensorOptions().dtype(k.dtype()).device(k.device()));
  torch::Tensor h_update =
      torch::empty({B, NT, H, K, K},
                   torch::TensorOptions().dtype(k.dtype()).device(k.device()));
  torch::Tensor final_state;
  if (output_final_state) {
    final_state = torch::empty({N, H, K, V},
                               torch::TensorOptions().dtype(torch::kFloat32).device(k.device()));
  }

  torch::Tensor v_new;
  if (save_new_value) {
    v_new = torch::empty_like(u);
  }

  auto npu_stream = c10_npu::getCurrentNPUStream(k_contig.device().index());
  rtStream_t stream = static_cast<rtStream_t>(npu_stream.stream());

  int32_t gridX = 1;
  int32_t gridY = static_cast<int32_t>(N * H);
  int32_t gridZ = 1;

  void* k_ptr = k_contig.data_ptr();
  void* w_ptr = w_contig.data_ptr();
  void* u_ptr = u_contig.data_ptr();
  void* v_new_ptr = save_new_value ? v_new.data_ptr() : nullptr;
  void* g_ptr = g_prepared.has_value() ? g_prepared.value().data_ptr() : nullptr;
  void* h_ptr = h.data_ptr();
  void* h0_ptr = initial_state_prepared.has_value()
                     ? initial_state_prepared.value().data_ptr()
                     : nullptr;
  void* ht_ptr = output_final_state ? final_state.data_ptr() : nullptr;
  void* cu_seqlens_ptr = cu_seqlens_prepared.has_value()
                             ? cu_seqlens_prepared.value().data_ptr()
                             : nullptr;
  void* chunk_offsets_ptr = chunk_offsets_prepared.has_value()
                                ? chunk_offsets_prepared.value().data_ptr()
                                : nullptr;
  void* h_update_ptr = h_update.data_ptr();

  auto& op = OperationFactory::instance().chunk_gated_delta_rule_fwd_h();
  auto ret = op.execute(stream, gridX, gridY, gridZ, [&](ArgsBuilder& ab) {
    ab.constructArgs(k_ptr,
                     u_ptr,
                     w_ptr,
                     v_new_ptr,
                     g_ptr,
                     h_ptr,
                     h0_ptr,
                     ht_ptr,
                     cu_seqlens_ptr,
                     chunk_offsets_ptr,
                     h_update_ptr,
                     static_cast<int32_t>(T),
                     static_cast<int32_t>(H),
                     static_cast<int32_t>(Hg),
                     static_cast<int32_t>(K),
                     static_cast<int32_t>(V));
  });

  if (ret != RT_ERROR_NONE) {
    LOG(ERROR) << "rtKernelLaunch failed for "
                  "'chunk_gated_delta_rule_fwd_kernel_h_blockdim64': "
               << ret;
  }

  record_tensor_if_needed(k_contig, npu_stream);
  record_tensor_if_needed(w_contig, npu_stream);
  record_tensor_if_needed(u_contig, npu_stream);
  if (g_prepared.has_value()) {
    record_tensor_if_needed(g_prepared.value(), npu_stream);
  }
  if (initial_state_prepared.has_value()) {
    record_tensor_if_needed(initial_state_prepared.value(), npu_stream);
  }
  if (cu_seqlens_prepared.has_value()) {
    record_tensor_if_needed(cu_seqlens_prepared.value(), npu_stream);
  }
  if (chunk_offsets_prepared.has_value()) {
    record_tensor_if_needed(chunk_offsets_prepared.value(), npu_stream);
  }
  record_tensor_if_needed(h_update, npu_stream);

  return std::make_tuple(h, v_new, final_state);
}

}  // namespace xllm::kernel::npu
