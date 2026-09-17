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

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <optional>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

constexpr float kOutputRtol = 0.005f;
constexpr float kOutputAtol = 0.01f;
constexpr float kStateRtol = 0.05f;
constexpr float kStateAtol = 0.01f;
constexpr int32_t kDeviceId = 0;

torch::Tensor l2norm(const torch::Tensor& x,
                     int64_t dim = -1,
                     float eps = 1e-6f) {
  auto inv_norm = torch::rsqrt((x * x).sum(dim, /*keepdim=*/true) + eps);
  return x * inv_norm;
}

// Direct port of Python recurrent_gated_delta_rule_ref (lines 385-419).
// Caller must pass pre-normalized q,k when use_qk_l2norm_in_kernel is true.
std::pair<torch::Tensor, torch::Tensor> torch_recurrent_gated_delta_rule(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& g,
    const torch::Tensor& beta,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    const std::optional<float>& scale_opt) {
  auto initial_dtype = query.scalar_type();
  auto device = query.device();

  // transpose(1,2): (B,T,H,K) -> (B,H,T,K), same as Python ref
  auto query_ = query.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto key_ = key.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto value_ = value.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto g_ = g.transpose(1, 2).contiguous().to(torch::kFloat32);
  auto beta_ = beta.transpose(1, 2).contiguous().to(torch::kFloat32);

  int64_t batch_size = key_.size(0);
  int64_t num_heads = key_.size(1);
  int64_t sequence_length = key_.size(2);
  int64_t k_head_dim = key_.size(3);
  int64_t v_head_dim = value_.size(3);

  // scale: default 1/sqrt(K) as in Python ref
  float scale = scale_opt.has_value()
                    ? scale_opt.value()
                    : 1.0f / std::sqrt(static_cast<float>(k_head_dim));
  query_ = query_ * scale;

  auto core_attn_out = torch::zeros(
      {batch_size, num_heads, sequence_length, v_head_dim}, value_.options());
  torch::Tensor h;

  if (initial_state.has_value() && initial_state.value().defined()) {
    h = initial_state.value().to(torch::kFloat32).to(device).contiguous();
  } else {
    h = torch::zeros({batch_size, num_heads, k_head_dim, v_head_dim},
                     value_.options());
  }

  for (int64_t i = 0; i < sequence_length; ++i) {
    auto b_q = query_.select(2, i);
    auto b_k = key_.select(2, i);
    auto b_v = value_.select(2, i).clone();
    auto b_g = g_.select(2, i).exp().unsqueeze(-1).unsqueeze(-1);
    auto b_beta = beta_.select(2, i).unsqueeze(-1);

    // h = h.clone() * g.exp()[..., None, None]
    h = h.clone() * b_g;
    // b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
    b_v = b_v - (h.clone() * b_k.unsqueeze(-1)).sum(-2);
    // b_v = b_v * b_beta[..., None]
    b_v = b_v * b_beta;
    // h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
    h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2);
    // o[:,:,i] = einsum("bhd,bhdm->bhm", b_q, h)
    core_attn_out.slice(2, i, i + 1) =
        (b_q.unsqueeze(-1) * h).sum(-2).unsqueeze(2);
  }

  if (!output_final_state) {
    h = torch::Tensor();
  }

  core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype);
  return std::make_pair(core_attn_out, h);
}

class TritonRecurrentGatedDeltaRuleTest : public ::testing::Test {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      c10_npu::SetDevice(kDeviceId);
      auto& reg = KernelRegistry::get_instance();
      std::string binary_path = GetKernelBinaryPath(
          "fused_recurrent_gated_delta_rule_fwd_kernel.npubin");
      std::string spec_binary_path = GetKernelBinaryPath(
          "fused_recurrent_gated_delta_rule_spec_fwd_kernel.npubin");
      npu_initialized_ =
          reg.register_kernel("fused_recurrent_gated_delta_rule_fwd_kernel",
                              binary_path) &&
          reg.get_kernel_stub("fused_recurrent_gated_delta_rule_fwd_kernel") !=
              nullptr &&
          reg.register_kernel("fused_recurrent_gated_delta_rule_spec_fwd_kernel",
                              spec_binary_path) &&
          reg.get_kernel_stub(
              "fused_recurrent_gated_delta_rule_spec_fwd_kernel") != nullptr;
    } catch (...) {
      npu_initialized_ = false;
    }
  }

  static void TearDownTestSuite() {
    if (npu_initialized_) {
      try {
        KernelRegistry::get_instance().cleanup();
      } catch (...) {
      }
    }
  }

  void SetUp() override {
    npu_available_ = npu_initialized_;
    if (!npu_available_) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU);
      return;
    }
    tensor_options_ = torch::TensorOptions()
                          .dtype(torch::kFloat16)
                          .device("npu:" + std::to_string(kDeviceId));
    torch::manual_seed(42);
    kernel_name_ = "fused_recurrent_gated_delta_rule_fwd_kernel";
    binary_filename_ = "fused_recurrent_gated_delta_rule_fwd_kernel.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
  }

  void TearDown() override {}

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

bool TritonRecurrentGatedDeltaRuleTest::npu_initialized_ = false;

// Test params aligned with Python test_accuracy_fused_recurrent:
// (B, T, H, HV, D, scale, gate_logit_normalizer, dtype)
struct RecurrentTestParam {
  int64_t B, T, H, HV, D;
  float scale, gate_logit_normalizer;
  torch::ScalarType dtype;
};

class TritonRecurrentGatedDeltaRuleParamTest
    : public TritonRecurrentGatedDeltaRuleTest,
      public ::testing::WithParamInterface<RecurrentTestParam> {};

TEST_P(TritonRecurrentGatedDeltaRuleParamTest, AccuracyMatch) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const auto& p = GetParam();
  auto device = at::Device(device_str_);
  const int64_t batch = p.B;
  const int64_t T = p.T;
  const int64_t num_heads = p.H;
  const int64_t num_v_heads = p.HV;
  const int64_t k_head_dim = p.D;
  const int64_t v_head_dim = p.D;
  const float scale_val = p.scale;
  const float gate_logit_normalizer = p.gate_logit_normalizer;
  constexpr bool use_qk_l2norm_in_kernel = true;

  torch::manual_seed(42);
  auto dtype = p.dtype;
  auto L = batch * T;

  // Input generation aligned with Python: beta.sigmoid(), g =
  // logsigmoid(rand)/gate_logit_normalizer
  auto q = torch::randn({batch, T, num_heads, k_head_dim}, dtype);
  auto k = torch::randn({batch, T, num_heads, k_head_dim}, dtype);
  auto v = torch::randn({batch, T, num_v_heads, v_head_dim}, dtype);
  auto beta = torch::rand({batch, T, num_v_heads}, dtype).sigmoid();
  auto g = torch::log_sigmoid(
               torch::rand({batch, T, num_v_heads}, torch::kFloat32)) /
           gate_logit_normalizer;
  auto initial_state = torch::randn(
      {batch, num_v_heads, k_head_dim, v_head_dim}, torch::kFloat32);

  torch::Tensor q_expanded = q, k_expanded = k;
  if (num_v_heads / num_heads > 1) {
    q_expanded = q.repeat_interleave(num_v_heads / num_heads, 2);
    k_expanded = k.repeat_interleave(num_v_heads / num_heads, 2);
  }

  // receives pre-normalized q,k; apply l2norm before calling
  if (use_qk_l2norm_in_kernel) {
    q_expanded = l2norm(q_expanded, -1, 1e-6f);
    k_expanded = l2norm(k_expanded, -1, 1e-6f);
  }

  auto [golden_o, golden_state] =
      torch_recurrent_gated_delta_rule(q_expanded,
                                       k_expanded,
                                       v,
                                       g,
                                       beta,
                                       initial_state,
                                       true,
                                       std::optional<float>(scale_val));

  auto q_d = q.reshape({1, L, num_heads, k_head_dim}).to(device);
  auto k_d = k.reshape({1, L, num_heads, k_head_dim}).to(device);
  auto v_d = v.reshape({1, L, num_v_heads, v_head_dim}).to(device);
  auto g_d = g.reshape({1, L, num_v_heads}).to(device);
  auto beta_d = beta.reshape({1, L, num_v_heads}).to(device);
  auto init_d = initial_state.clone().to(device);

  std::vector<int64_t> culen;
  culen.reserve(batch + 1);
  for (int i = 0; i <= batch; ++i) {
    culen.push_back(i * T);
  }
  auto cu_seqlens = torch::tensor(culen, torch::kInt64).to(device);
  auto ssm_state_indices = torch::arange(
      batch, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto [o_d, state_d] =
      npu_fused_recurrent_gated_delta_rule(q_d,
                                           k_d,
                                           v_d,
                                           g_d,
                                           beta_d,
                                           scale_val,
                                           init_d,
                                           true,
                                           cu_seqlens,
                                           ssm_state_indices,
                                           std::nullopt,
                                           use_qk_l2norm_in_kernel);
  aclrtSynchronizeStream(npu_stream.stream());

  auto o = o_d.cpu().reshape(golden_o.sizes());
  auto state = state_d.cpu();

  // rtol=0.005, atol=0.01 for output
  auto output_diff =
      (golden_o.to(torch::kFloat32) - o.to(torch::kFloat32)).abs();
  auto output_max_diff = output_diff.max().item<float>();
  EXPECT_LT(output_max_diff, kOutputAtol)
      << "Output mismatch: max diff = " << output_max_diff
      << ", shape: " << o.sizes();

  // rtol=0.05, atol=0.01 for state
  auto state_diff =
      (golden_state.to(torch::kFloat32) - state.to(torch::kFloat32)).abs();
  auto state_max_diff = state_diff.max().item<float>();
  EXPECT_LT(state_max_diff, kStateAtol)
      << "State mismatch: max diff = " << state_max_diff
      << ", shape: " << state.sizes();
}

INSTANTIATE_TEST_SUITE_P(
    RecurrentParams,
    TritonRecurrentGatedDeltaRuleParamTest,
    ::testing::Values(
        RecurrentTestParam{1, 1, 4, 8, 128, 0.5f, 0.1f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 4, 8, 128, 0.5f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{4, 1, 4, 8, 128, 0.2f, 0.1f, torch::kBFloat16},
        RecurrentTestParam{8, 1, 4, 8, 128, 0.3f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{16, 1, 4, 8, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{32, 1, 4, 8, 128, 0.2f, 1.0f, torch::kBFloat16},
        // Qwen3.5/Qwen3.6 local GDN shapes for TP1/2/4/8.
        RecurrentTestParam{2, 1, 16, 16, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 8, 8, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 4, 4, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 2, 2, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 16, 32, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 8, 16, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 2, 4, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 16, 48, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 8, 24, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 4, 12, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 2, 6, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 16, 64, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 8, 32, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 4, 16, 128, 0.2f, 1.0f, torch::kBFloat16},
        RecurrentTestParam{2, 1, 2, 8, 128, 0.2f, 1.0f, torch::kBFloat16},
        // Regression for the previously observed TP4 spec-local shape.
        RecurrentTestParam{2, 1, 1, 8, 128, 0.2f, 1.0f, torch::kBFloat16}));

TEST_F(TritonRecurrentGatedDeltaRuleTest, SpecAccuracyAcceptedOffsets) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t num_sequences = 3;
  constexpr int64_t T = 4;
  constexpr int64_t num_heads = 8;
  constexpr int64_t num_v_heads = 16;
  constexpr int64_t head_dim = 128;
  constexpr float scale_val = 0.5f;
  constexpr bool use_qk_l2norm_in_kernel = true;
  const auto dtype = torch::kBFloat16;
  const int64_t total_tokens = num_sequences * T;

  torch::manual_seed(44);
  auto q = torch::randn({num_sequences, T, num_heads, head_dim}, dtype);
  auto k = torch::randn({num_sequences, T, num_heads, head_dim}, dtype);
  auto v = torch::randn({num_sequences, T, num_v_heads, head_dim}, dtype);
  auto beta = torch::rand({num_sequences, T, num_v_heads}, dtype).sigmoid();
  auto g =
      torch::log_sigmoid(torch::rand({num_sequences, T, num_v_heads},
                                     torch::kFloat32));
  auto initial_state =
      torch::randn({32, num_v_heads, head_dim, head_dim}, torch::kFloat32);
  auto state_indices =
      torch::tensor({{2, 4, 6, 8}, {11, 13, 15, 17}, {20, 22, 24, 26}},
                    torch::kInt32);
  auto num_accepted_tokens = torch::tensor({1, 2, 3}, torch::kInt32);

  auto q_d = q.reshape({1, total_tokens, num_heads, head_dim}).to(device);
  auto k_d = k.reshape({1, total_tokens, num_heads, head_dim}).to(device);
  auto v_d = v.reshape({1, total_tokens, num_v_heads, head_dim}).to(device);
  auto g_d = g.reshape({1, total_tokens, num_v_heads}).to(device);
  auto beta_d = beta.reshape({1, total_tokens, num_v_heads}).to(device);
  auto init_d = initial_state.clone().to(device);
  auto state_indices_d = state_indices.to(device);
  auto num_accepted_tokens_d = num_accepted_tokens.to(device);

  auto cu_seqlens =
      torch::arange(num_sequences + 1,
                    torch::TensorOptions().dtype(torch::kInt64).device(device)) *
      T;

  std::vector<torch::Tensor> ref_chunks;
  std::vector<torch::Tensor> ref_state_chunks;
  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    auto q_i = q_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto k_i = k_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto v_i = v_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto g_i = g_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto beta_i = beta_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto state_idx =
        state_indices.index({seq_idx, num_accepted_tokens[seq_idx].item<int>() - 1})
            .item<int>();
    auto h = initial_state.index({state_idx}).unsqueeze(0);
    auto q_i_expanded = q_i.repeat_interleave(num_v_heads / num_heads, 2);
    auto k_i_expanded = k_i.repeat_interleave(num_v_heads / num_heads, 2);
    if (use_qk_l2norm_in_kernel) {
      q_i_expanded = l2norm(q_i_expanded, -1, 1e-6f).to(dtype);
      k_i_expanded = l2norm(k_i_expanded, -1, 1e-6f).to(dtype);
    }

    std::vector<torch::Tensor> seq_outputs;
    std::vector<torch::Tensor> seq_states;
    for (int64_t token_idx = 0; token_idx < T; ++token_idx) {
      auto [ref_i, h_next] =
          torch_recurrent_gated_delta_rule(q_i_expanded.slice(1, token_idx, token_idx + 1),
                                           k_i_expanded.slice(1, token_idx, token_idx + 1),
                                           v_i.slice(1, token_idx, token_idx + 1),
                                           g_i.slice(1, token_idx, token_idx + 1),
                                           beta_i.slice(1, token_idx, token_idx + 1),
                                           h,
                                           true,
                                           std::optional<float>(scale_val));
      seq_outputs.emplace_back(ref_i);
      seq_states.emplace_back(h_next.squeeze(0));
      h = h_next;
    }
    ref_chunks.emplace_back(torch::cat(seq_outputs, 1));
    ref_state_chunks.emplace_back(torch::stack(seq_states, 0));
  }
  auto golden_o =
      torch::cat(ref_chunks, 1).reshape({num_sequences, T, num_v_heads, head_dim});
  auto golden_states = torch::cat(ref_state_chunks, 0);

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto [o_d, state_d] =
      npu_fused_recurrent_gated_delta_rule(q_d,
                                           k_d,
                                           v_d,
                                           g_d,
                                           beta_d,
                                           scale_val,
                                           init_d,
                                           true,
                                           cu_seqlens,
                                           state_indices_d,
                                           num_accepted_tokens_d,
                                           use_qk_l2norm_in_kernel);
  aclrtSynchronizeStream(npu_stream.stream());

  auto o = o_d.cpu().reshape(golden_o.sizes());
  auto flat_state_indices = state_indices.reshape({-1}).to(torch::kLong);
  auto states = state_d.cpu().index_select(0, flat_state_indices);

  auto output_diff =
      (golden_o.to(torch::kFloat32) - o.to(torch::kFloat32)).abs();
  EXPECT_LT(output_diff.max().item<float>(), kOutputAtol)
      << "Spec output mismatch: max diff = "
      << output_diff.max().item<float>();

  auto state_diff =
      (golden_states.to(torch::kFloat32) - states.to(torch::kFloat32)).abs();
  EXPECT_LT(state_diff.max().item<float>(), kStateAtol)
      << "Spec state mismatch: max diff = " << state_diff.max().item<float>();
}

TEST_F(TritonRecurrentGatedDeltaRuleTest, SpecAccuracyTp4LocalShape) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t num_sequences = 8;
  constexpr int64_t T = 2;
  constexpr int64_t num_heads = 1;
  constexpr int64_t num_v_heads = 8;
  constexpr int64_t head_dim = 128;
  constexpr float scale_val = 0.08838834764831845f;  // 1 / sqrt(128)
  constexpr bool use_qk_l2norm_in_kernel = true;
  const auto dtype = torch::kBFloat16;
  const int64_t total_tokens = num_sequences * T;

  torch::manual_seed(45);
  auto q = torch::randn({num_sequences, T, num_heads, head_dim}, dtype);
  auto k = torch::randn({num_sequences, T, num_heads, head_dim}, dtype);
  auto v = torch::randn({num_sequences, T, num_v_heads, head_dim}, dtype);
  auto beta = torch::rand({num_sequences, T, num_v_heads}, dtype).sigmoid();
  auto g =
      torch::log_sigmoid(torch::rand({num_sequences, T, num_v_heads},
                                     torch::kFloat32));
  auto initial_state =
      torch::randn({64, num_v_heads, head_dim, head_dim}, torch::kFloat32);

  auto state_indices =
      (torch::arange(num_sequences * T, torch::kInt32).reshape({num_sequences, T}) +
       8);
  auto num_accepted_tokens =
      torch::tensor({1, 2, 1, 2, 1, 2, 1, 2}, torch::kInt32);

  auto q_d = q.reshape({1, total_tokens, num_heads, head_dim}).to(device);
  auto k_d = k.reshape({1, total_tokens, num_heads, head_dim}).to(device);
  auto v_d = v.reshape({1, total_tokens, num_v_heads, head_dim}).to(device);
  auto g_d = g.reshape({1, total_tokens, num_v_heads}).to(device);
  auto beta_d = beta.reshape({1, total_tokens, num_v_heads}).to(device);
  auto init_d = initial_state.clone().to(device);
  auto state_indices_d = state_indices.to(device);
  auto num_accepted_tokens_d = num_accepted_tokens.to(device);

  auto cu_seqlens =
      torch::arange(num_sequences + 1,
                    torch::TensorOptions().dtype(torch::kInt64).device(device)) *
      T;

  std::vector<torch::Tensor> ref_chunks;
  std::vector<torch::Tensor> ref_state_chunks;
  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    auto q_i = q_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto k_i = k_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto v_i = v_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto g_i = g_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto beta_i = beta_d.slice(1, seq_idx * T, (seq_idx + 1) * T).cpu();
    auto accepted_idx = num_accepted_tokens[seq_idx].item<int>() - 1;
    auto state_idx = state_indices.index({seq_idx, accepted_idx}).item<int>();
    auto h = initial_state.index({state_idx}).unsqueeze(0);
    auto q_i_expanded = q_i.repeat_interleave(num_v_heads / num_heads, 2);
    auto k_i_expanded = k_i.repeat_interleave(num_v_heads / num_heads, 2);
    if (use_qk_l2norm_in_kernel) {
      q_i_expanded = l2norm(q_i_expanded, -1, 1e-6f).to(dtype);
      k_i_expanded = l2norm(k_i_expanded, -1, 1e-6f).to(dtype);
    }

    std::vector<torch::Tensor> seq_outputs;
    std::vector<torch::Tensor> seq_states;
    for (int64_t token_idx = 0; token_idx < T; ++token_idx) {
      auto [ref_i, h_next] =
          torch_recurrent_gated_delta_rule(q_i_expanded.slice(1, token_idx, token_idx + 1),
                                           k_i_expanded.slice(1, token_idx, token_idx + 1),
                                           v_i.slice(1, token_idx, token_idx + 1),
                                           g_i.slice(1, token_idx, token_idx + 1),
                                           beta_i.slice(1, token_idx, token_idx + 1),
                                           h,
                                           true,
                                           std::optional<float>(scale_val));
      seq_outputs.emplace_back(ref_i);
      seq_states.emplace_back(h_next.squeeze(0));
      h = h_next;
    }
    ref_chunks.emplace_back(torch::cat(seq_outputs, 1));
    ref_state_chunks.emplace_back(torch::stack(seq_states, 0));
  }
  auto golden_o =
      torch::cat(ref_chunks, 1).reshape({num_sequences, T, num_v_heads, head_dim});
  auto golden_states = torch::cat(ref_state_chunks, 0);

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto [o_d, state_d] =
      npu_fused_recurrent_gated_delta_rule(q_d,
                                           k_d,
                                           v_d,
                                           g_d,
                                           beta_d,
                                           scale_val,
                                           init_d,
                                           true,
                                           cu_seqlens,
                                           state_indices_d,
                                           num_accepted_tokens_d,
                                           use_qk_l2norm_in_kernel);
  aclrtSynchronizeStream(npu_stream.stream());

  auto o = o_d.cpu().reshape(golden_o.sizes());
  auto flat_state_indices = state_indices.reshape({-1}).to(torch::kLong);
  auto states = state_d.cpu().index_select(0, flat_state_indices);

  auto output_diff =
      (golden_o.to(torch::kFloat32) - o.to(torch::kFloat32)).abs();
  EXPECT_LT(output_diff.max().item<float>(), kOutputAtol)
      << "TP4-local spec output mismatch: max diff = "
      << output_diff.max().item<float>();

  auto state_diff =
      (golden_states.to(torch::kFloat32) - states.to(torch::kFloat32)).abs();
  EXPECT_LT(state_diff.max().item<float>(), kStateAtol)
      << "TP4-local spec state mismatch: max diff = "
      << state_diff.max().item<float>();
}

}  // namespace xllm::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  xllm::testing::init_npu_test_runtime();

  bool npu_available = false;
  std::string device_str =
      "npu:" + std::to_string(xllm::kernel::npu::kDeviceId);
  try {
    auto test_tensor =
        torch::zeros({1}, torch::TensorOptions().device(device_str));
    (void)test_tensor;
    npu_available = true;
  } catch (...) {
    npu_available = false;
  }

  if (!npu_available) {
    LOG(WARNING) << "NPU device not available, skipping all tests.";
    return 0;
  }

  int result = RUN_ALL_TESTS();
  return result;
}
