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
 *
 * C++ unit-test for npu_chunk_gated_delta_rule.
 *
 * The test follows the same pattern as other Triton adapter tests:
 *   1. Build a CPU golden implementation for the full chunk_gated_delta_rule
 *      forward path.
 *   2. Run the exported torch_api entry on NPU with DFQ real-world shapes.
 *   3. Compare output / final_state numerically within tolerance.
 */

#include <acl/acl.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <optional>
#include <string>
#include <vector>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

constexpr int32_t kDeviceId = 0;
constexpr auto kKernelName = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64";
constexpr auto kBinaryName =
    "chunk_gated_delta_rule_fwd_kernel_h_blockdim64.npubin";
constexpr float kCloseAtol = 1e-2f;
constexpr float kCloseRtol = 1e-2f;

using torch::indexing::Slice;

struct ChunkTestParam {
  std::string name;
  std::vector<int64_t> q_shape;
  std::vector<int64_t> v_shape;
  std::vector<int64_t> initial_state_shape;
  int64_t chunk_size;
  std::vector<int32_t> cu_seqlens;
  torch::ScalarType dtype;
  torch::ScalarType initial_state_dtype;
  bool create_inputs_on_cpu;
  bool use_qk_l2norm_in_kernel;
};

ChunkTestParam MakePythonVarlenCase(std::string name,
                                    std::vector<int64_t> q_shape,
                                    std::vector<int64_t> v_shape,
                                    std::vector<int64_t> initial_state_shape,
                                    std::vector<int32_t> cu_seqlens,
                                    bool create_inputs_on_cpu,
                                    torch::ScalarType initial_state_dtype =
                                        torch::kBFloat16) {
  return ChunkTestParam{
      std::move(name),
      std::move(q_shape),
      std::move(v_shape),
      std::move(initial_state_shape),
      64,
      std::move(cu_seqlens),
      torch::kBFloat16,
      initial_state_dtype,
      create_inputs_on_cpu,
      true,
  };
}

ChunkTestParam MakeRealShapeCase(std::string name,
                                    bool create_inputs_on_cpu,
                                    torch::ScalarType initial_state_dtype =
                                        torch::kBFloat16) {
  return MakePythonVarlenCase(std::move(name),
                              {1, 31, 4, 128},
                              {1, 31, 8, 128},
                              {1, 8, 128, 128},
                              {0, 31},
                              create_inputs_on_cpu,
                              initial_state_dtype);
}

ChunkTestParam MakeLongRealShapeCase(std::string name,
                                     bool create_inputs_on_cpu,
                                     torch::ScalarType initial_state_dtype =
                                         torch::kBFloat16) {
  return MakePythonVarlenCase(std::move(name),
                              {1, 10240, 16, 128},
                              {1, 10240, 16, 128},
                              {1, 16, 128, 128},
                              {0, 10240},
                              create_inputs_on_cpu,
                              initial_state_dtype);
}

ChunkTestParam MakeRequested4096GroupedHeadsCase(
    std::string name,
    int64_t num_heads_qk,
    int64_t num_heads_v,
    bool create_inputs_on_cpu,
    torch::ScalarType initial_state_dtype = torch::kBFloat16) {
  return MakePythonVarlenCase(std::move(name),
                              {1, 4096, num_heads_qk, 128},
                              {1, 4096, num_heads_v, 128},
                              {1, num_heads_v, 128, 128},
                              {0, 4096},
                              create_inputs_on_cpu,
                              initial_state_dtype);
}

struct ChunkRefResult {
  torch::Tensor out;
  torch::Tensor final_state;
};

std::vector<int64_t> BuildChunkOffsetsHost(const std::vector<int32_t>& cu_seqlens,
                                           int64_t chunk_size) {
  std::vector<int64_t> chunk_offsets;
  chunk_offsets.reserve(cu_seqlens.size());
  chunk_offsets.push_back(0);
  for (size_t i = 0; i + 1 < cu_seqlens.size(); ++i) {
    const int64_t len =
        static_cast<int64_t>(cu_seqlens[i + 1]) - static_cast<int64_t>(cu_seqlens[i]);
    chunk_offsets.push_back(chunk_offsets.back() +
                            (len + chunk_size - 1) / chunk_size);
  }
  return chunk_offsets;
}

torch::Tensor ExpandQkToVHeads(const torch::Tensor& x, int64_t num_heads_v) {
  const int64_t num_heads_qk = x.size(2);
  TORCH_CHECK(num_heads_v % num_heads_qk == 0,
              "num_heads_v must be divisible by num_heads_qk.");
  if (num_heads_v == num_heads_qk) {
    return x.contiguous();
  }
  const int64_t group_size = num_heads_v / num_heads_qk;
  return x.unsqueeze(3)
      .expand({x.size(0), x.size(1), num_heads_qk, group_size, x.size(3)})
      .reshape({x.size(0), x.size(1), num_heads_v, x.size(3)})
      .contiguous();
}

torch::Tensor BuildChunkLocalCumsum(const torch::Tensor& g_cpu,
                                    const std::vector<int32_t>& cu_seqlens,
                                    int64_t chunk_size) {
  TORCH_CHECK(!g_cpu.is_cuda(), "Expected CPU tensor for BuildChunkLocalCumsum.");
  auto g_f32 = g_cpu.to(torch::kFloat32);
  auto out = torch::zeros_like(g_f32);

  const int64_t num_sequences = static_cast<int64_t>(cu_seqlens.size()) - 1;
  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const int64_t start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    for (int64_t token_start = start; token_start < end; token_start += chunk_size) {
      const int64_t token_len = std::min<int64_t>(chunk_size, end - token_start);
      auto g_chunk = g_f32.index({0, Slice(token_start, token_start + token_len), Slice()});
      out.index_put_(
          {0, Slice(token_start, token_start + token_len), Slice()},
          torch::cumsum(g_chunk, /*dim=*/0));
    }
  }
  return out;
}

torch::Tensor CpuL2NormLastDim(const torch::Tensor& x_cpu,
                               double eps = 1e-6) {
  TORCH_CHECK(!x_cpu.is_cuda(), "CpuL2NormLastDim expects a CPU tensor.");
  auto x_f32 = x_cpu.to(torch::kFloat32);
  auto denom = torch::sqrt(torch::sum(x_f32 * x_f32, -1, true) + eps);
  return x_f32 / denom;
}

ChunkRefResult chunk_gated_delta_rule_full_op_ref(
    const torch::Tensor& q_cpu_raw,
    const torch::Tensor& k_cpu_raw,
    const torch::Tensor& v_cpu_raw,
    const torch::Tensor& g_cpu_raw,
    const torch::Tensor& beta_cpu_raw,
    const torch::Tensor& initial_state_cpu,
    const std::vector<int32_t>& cu_seqlens,
    int64_t chunk_size,
    bool use_qk_l2norm_in_kernel,
    torch::ScalarType input_dtype) {
  TORCH_CHECK(!q_cpu_raw.is_cuda() && !k_cpu_raw.is_cuda() && !v_cpu_raw.is_cuda() &&
                  !g_cpu_raw.is_cuda() && !beta_cpu_raw.is_cuda() &&
                  !initial_state_cpu.is_cuda(),
              "Reference path expects CPU tensors.");

  const int64_t batch_size = k_cpu_raw.size(0);
  const int64_t num_heads_qk = k_cpu_raw.size(2);
  const int64_t head_dim_k = k_cpu_raw.size(3);
  const int64_t num_heads_v = v_cpu_raw.size(2);
  const int64_t head_dim_v = v_cpu_raw.size(3);
  const int64_t num_sequences =
      cu_seqlens.empty() ? batch_size : static_cast<int64_t>(cu_seqlens.size()) - 1;

  TORCH_CHECK(batch_size == 1,
              "Current CPU reference path only supports B=1 varlen inputs.");
  TORCH_CHECK(num_heads_v % num_heads_qk == 0,
              "num_heads_v must be divisible by num_heads_qk.");

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_k));
  auto q_norm_input = q_cpu_raw.contiguous();
  auto k_norm_input = k_cpu_raw.contiguous();
  auto q_full = use_qk_l2norm_in_kernel ? CpuL2NormLastDim(q_norm_input)
                                        : q_cpu_raw;
  auto k_full = use_qk_l2norm_in_kernel ? CpuL2NormLastDim(k_norm_input)
                                        : k_cpu_raw;
  auto v_full = v_cpu_raw.to(torch::kFloat32);
  auto beta_full = beta_cpu_raw.to(torch::kFloat32);
  auto g_cumsum = BuildChunkLocalCumsum(g_cpu_raw, cu_seqlens, chunk_size);
  auto q_expanded = ExpandQkToVHeads(q_full, num_heads_v);
  auto k_expanded = ExpandQkToVHeads(k_full, num_heads_v);

  auto out_ref = torch::zeros(
      {batch_size, k_cpu_raw.size(1), num_heads_v, head_dim_v},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto final_state_ref = torch::empty(
      {num_sequences, num_heads_v, head_dim_k, head_dim_v},
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

  auto q_b = q_expanded.select(0, 0);
  auto k_b = k_expanded.select(0, 0);
  auto v_b = v_full.select(0, 0);
  auto beta_b = beta_full.select(0, 0);
  auto g_b = g_cumsum.select(0, 0);

  for (int64_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const int64_t start = static_cast<int64_t>(cu_seqlens[seq_idx]);
    const int64_t end = static_cast<int64_t>(cu_seqlens[seq_idx + 1]);
    const int64_t seq_len = end - start;
    const int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;

    for (int64_t head_idx = 0; head_idx < num_heads_v; ++head_idx) {
      auto state = initial_state_cpu.select(0, seq_idx)
                       .select(0, head_idx)
                       .to(torch::kFloat32)
                       .clone();

      for (int64_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
        const int64_t token_start = start + chunk_idx * chunk_size;
        const int64_t token_len =
            std::min<int64_t>(chunk_size, end - token_start);

        auto q_chunk = q_b.narrow(0, token_start, token_len)
                           .select(1, head_idx)
                           .to(torch::kFloat32) * scale;
        auto k_chunk = k_b.narrow(0, token_start, token_len)
                           .select(1, head_idx)
                           .to(torch::kFloat32);
        auto v_chunk = v_b.narrow(0, token_start, token_len)
                           .select(1, head_idx)
                           .to(torch::kFloat32);
        auto beta_chunk = beta_b.narrow(0, token_start, token_len)
                              .select(1, head_idx)
                              .to(torch::kFloat32);
        auto g_chunk = g_b.narrow(0, token_start, token_len).select(1, head_idx);

        auto k_beta = k_chunk * beta_chunk.unsqueeze(-1);
        auto g_diff = g_chunk.unsqueeze(1) - g_chunk.unsqueeze(0);
        auto decay_mask = torch::tril(torch::exp(g_diff));
        auto solver =
            -torch::matmul(k_beta, k_chunk.transpose(0, 1)) * decay_mask;
        solver = torch::tril(solver, -1);
        for (int64_t row_idx = 1; row_idx < token_len; ++row_idx) {
          auto row = solver.index({row_idx, Slice(0, row_idx)}).clone().unsqueeze(0);
          auto sub = solver.index({Slice(0, row_idx), Slice(0, row_idx)}).clone();
          solver.index_put_(
              {row_idx, Slice(0, row_idx)},
              (row + torch::matmul(row, sub)).squeeze(0));
        }
        solver +=
            torch::eye(token_len, torch::TensorOptions().dtype(torch::kFloat32));

        auto v_beta = v_chunk * beta_chunk.unsqueeze(-1);
        auto value_chunk = torch::matmul(solver, v_beta);
        auto k_cumdecay = torch::matmul(
            solver,
            k_chunk * beta_chunk.unsqueeze(-1) * torch::exp(g_chunk).unsqueeze(-1));
        auto attn_inter =
            torch::matmul(q_chunk, k_chunk.transpose(0, 1)) * decay_mask;
        attn_inter = torch::tril(attn_inter, 0);
        auto v_prime = torch::matmul(k_cumdecay, state);
        auto v_new_chunk = value_chunk - v_prime;
        auto inter_state =
            torch::matmul(q_chunk * torch::exp(g_chunk).unsqueeze(-1), state);
        auto out_chunk = inter_state + torch::matmul(attn_inter, v_new_chunk);
        out_ref.index_put_(
            {0, Slice(token_start, token_start + token_len), head_idx, Slice()},
            out_chunk);

        const float last_g = g_chunk.index({token_len - 1}).item<float>();
        auto decay =
            torch::exp(torch::full_like(g_chunk, last_g) - g_chunk).unsqueeze(-1);
        auto k_decay = k_chunk * decay;

        state = state * std::exp(last_g) +
                torch::matmul(k_decay.transpose(0, 1), v_new_chunk);
      }

      final_state_ref.index_put_({seq_idx, head_idx}, state);
    }
  }

  return {out_ref.to(input_dtype), final_state_ref};
}

class TritonChunkGatedDeltaRuleTest
    : public ::testing::TestWithParam<ChunkTestParam> {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
      auto& reg = KernelRegistry::get_instance();
      std::string binary_path = GetKernelBinaryPath(kBinaryName);
      npu_initialized_ = reg.register_kernel(kKernelName, binary_path) &&
                         reg.get_kernel_stub(kKernelName) != nullptr;
    } catch (...) {
      npu_initialized_ = false;
    }
  }

  static void TearDownTestSuite() {
    if (!npu_initialized_) {
      return;
    }
    try {
      KernelRegistry::get_instance().cleanup();
      torch_npu::finalize_npu();
    } catch (...) {
    }
  }

  void SetUp() override {
    npu_available_ = npu_initialized_;
    device_str_ = "npu:" + std::to_string(kDeviceId);
    tensor_options_ =
        torch::TensorOptions().dtype(GetParam().dtype).device(device_str_);
  }

  void TearDown() override {
    if (!npu_available_) {
      return;
    }
    try {
      auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
      aclrtSynchronizeStream(npu_stream.stream());
    } catch (...) {
    }
    try {
      c10_npu::NPUCachingAllocator::emptyCache(false);
    } catch (...) {
    }
  }

  bool npu_available_ = false;
  std::string device_str_;
  torch::TensorOptions tensor_options_;
};

bool TritonChunkGatedDeltaRuleTest::npu_initialized_ = false;

TEST_P(TritonChunkGatedDeltaRuleTest, FullForwardAgainstCpuReference) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const auto& p = GetParam();
  SCOPED_TRACE("case=" + p.name);
  torch::manual_seed(42);

  auto device = at::Device(device_str_);
  auto opts_cpu_data =
      torch::TensorOptions().dtype(p.dtype).device(torch::kCPU);
  auto opts_cpu_gate =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto opts_cpu_state =
      torch::TensorOptions().dtype(p.initial_state_dtype).device(torch::kCPU);

  auto input_opts = p.create_inputs_on_cpu ? opts_cpu_data : tensor_options_;
  auto gate_opts = p.create_inputs_on_cpu
                       ? opts_cpu_gate
                       : opts_cpu_gate.device(device_str_);
  auto state_opts = p.create_inputs_on_cpu
                        ? opts_cpu_state
                        : opts_cpu_state.device(device_str_);
  const int64_t batch_size = p.q_shape.at(0);
  const int64_t seq_len = p.q_shape.at(1);
  const int64_t num_heads_qk = p.q_shape.at(2);
  const int64_t head_dim_k = p.q_shape.at(3);
  const int64_t num_heads_v = p.v_shape.at(2);
  const int64_t head_dim_v = p.v_shape.at(3);

  auto q_raw = torch::randn(p.q_shape, input_opts);
  auto k_raw = torch::randn(p.q_shape, input_opts);
  auto v_raw = torch::randn(p.v_shape, input_opts);
  auto gate_shape =
      std::vector<int64_t>{p.v_shape.at(0), p.v_shape.at(1), p.v_shape.at(2)};
  auto g_raw = torch::log(torch::sigmoid(torch::randn(gate_shape, gate_opts)));
  auto beta_raw = torch::sigmoid(torch::randn(gate_shape, gate_opts)).to(p.dtype);
  auto initial_state = torch::randn(p.initial_state_shape, state_opts);
  auto cu_seqlens_opts = torch::TensorOptions().dtype(torch::kInt32);
  if (p.create_inputs_on_cpu) {
    cu_seqlens_opts = cu_seqlens_opts.device(torch::kCPU);
  } else {
    cu_seqlens_opts = cu_seqlens_opts.device(device);
  }
  auto cu_seqlens = torch::tensor(p.cu_seqlens, cu_seqlens_opts);

  auto k_cpu = p.create_inputs_on_cpu ? k_raw : k_raw.cpu();
  auto v_cpu = p.create_inputs_on_cpu ? v_raw : v_raw.cpu();
  auto g_cpu = p.create_inputs_on_cpu ? g_raw : g_raw.cpu();
  auto beta_cpu = p.create_inputs_on_cpu ? beta_raw : beta_raw.cpu();
  auto initial_state_cpu =
      p.create_inputs_on_cpu ? initial_state : initial_state.cpu();
  auto initial_state_ref = initial_state_cpu.to(torch::kFloat32);

  auto q_cpu = p.create_inputs_on_cpu ? q_raw : q_raw.cpu();

  auto ref = chunk_gated_delta_rule_full_op_ref(
      q_cpu,
      k_cpu,
      v_cpu,
      g_cpu,
      beta_cpu,
      initial_state_ref,
      p.cu_seqlens,
      p.chunk_size,
      p.use_qk_l2norm_in_kernel,
      p.dtype);

  auto q_d = p.create_inputs_on_cpu ? q_raw.to(device) : q_raw;
  auto k_d = p.create_inputs_on_cpu ? k_raw.to(device) : k_raw;
  auto v_d = p.create_inputs_on_cpu ? v_raw.to(device) : v_raw;
  auto g_d = p.create_inputs_on_cpu ? g_raw.to(device) : g_raw;
  auto beta_d = p.create_inputs_on_cpu ? beta_raw.to(device) : beta_raw;
  auto init_d = p.create_inputs_on_cpu ? initial_state.to(device) : initial_state;
  auto cu_seqlens_d = p.create_inputs_on_cpu ? cu_seqlens.to(device) : cu_seqlens;

  auto [out_d, final_state_d] = npu_chunk_gated_delta_rule(q_d,
                                                           k_d,
                                                           v_d,
                                                           g_d,
                                                           beta_d,
                                                           std::nullopt,
                                                           init_d,
                                                           true,
                                                           cu_seqlens_d,
                                                           false,
                                                           p.use_qk_l2norm_in_kernel);

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  auto sync_ret = aclrtSynchronizeStream(npu_stream.stream());
  ASSERT_EQ(sync_ret, ACL_ERROR_NONE) << "aclrtSynchronizeStream failed with "
                                      << sync_ret;

  EXPECT_TRUE(out_d.defined());
  EXPECT_TRUE(final_state_d.defined());
  EXPECT_EQ(out_d.device().type(), at::kPrivateUse1);
  EXPECT_EQ(final_state_d.device().type(), at::kPrivateUse1);

  const int64_t num_sequences = static_cast<int64_t>(p.cu_seqlens.size()) - 1;
  EXPECT_EQ(out_d.sizes(),
            std::vector<int64_t>({batch_size, seq_len, num_heads_v, head_dim_v}));
  EXPECT_EQ(final_state_d.sizes(),
            std::vector<int64_t>({num_sequences, num_heads_v, head_dim_k, head_dim_v}));

  auto check_close = [&](const torch::Tensor& golden,
                         const torch::Tensor& actual,
                         const char* name) {
    auto golden_f32 = golden.to(torch::kFloat32).contiguous();
    auto actual_f32 = actual.cpu().contiguous().to(torch::kFloat32);
    const bool is_close =
        torch::allclose(golden_f32, actual_f32, kCloseRtol, kCloseAtol, false);
    auto diff = torch::abs(golden_f32 - actual_f32);
    const float max_diff = diff.max().item<float>();
    EXPECT_TRUE(is_close)
        << name << " mismatch: max_diff=" << max_diff
        << ", atol=" << kCloseAtol
        << ", rtol=" << kCloseRtol
        << " (case=" << p.name << ")";
  };

  check_close(ref.out, out_d, "out");
  check_close(ref.final_state, final_state_d, "final_state");
}

INSTANTIATE_TEST_SUITE_P(
    ChunkGatedDeltaRuleCases,
    TritonChunkGatedDeltaRuleTest,
    ::testing::Values(
        MakeRealShapeCase("RealShapeDeviceInput", false),
        MakeRealShapeCase("RealShapeCpuInput", true),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To16DeviceInput",
                                          8,
                                          16,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To16CpuInput",
                                          8,
                                          16,
                                          true),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To16DeviceInputFp32State",
                                          8,
                                          16,
                                          false,
                                          torch::kFloat32),
        MakeRequested4096GroupedHeadsCase("Requested4096H16To32DeviceInput",
                                          16,
                                          32,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H16To32DeviceInputFp32State",
                                          16,
                                          32,
                                          false,
                                          torch::kFloat32),
        MakeRequested4096GroupedHeadsCase("Requested4096H4To8DeviceInput",
                                          4,
                                          8,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H4To8DeviceInputFp32State",
                                          4,
                                          8,
                                          false,
                                          torch::kFloat32),
        MakeRequested4096GroupedHeadsCase("Requested4096H16To16DeviceInput",
                                          16,
                                          16,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To8DeviceInput",
                                          8,
                                          8,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H4To4DeviceInput",
                                          4,
                                          4,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H2To2DeviceInput",
                                          2,
                                          2,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H16To48DeviceInput",
                                          16,
                                          48,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To24DeviceInput",
                                          8,
                                          24,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H4To12DeviceInput",
                                          4,
                                          12,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H2To6DeviceInput",
                                          2,
                                          6,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H16To64DeviceInput",
                                          16,
                                          64,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H8To32DeviceInput",
                                          8,
                                          32,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H4To16DeviceInput",
                                          4,
                                          16,
                                          false),
        MakeRequested4096GroupedHeadsCase("Requested4096H2To8DeviceInput",
                                          2,
                                          8,
                                          false),
        MakeLongRealShapeCase("LongRealShapeDeviceInput", false),
        MakeRealShapeCase(
            "RealShapeDeviceInputFloatInitialState",
            false,
            torch::kFloat32)),
    [](const ::testing::TestParamInfo<ChunkTestParam>& info) {
      return info.param.name;
    });

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

  return RUN_ALL_TESTS();
}
