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

#include <algorithm>
#include <optional>
#include <vector>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

constexpr float kToleranceV2 = 5e-2f;
constexpr int32_t kDeviceIdV2 = 0;
bool g_npu_initialized_v2 = false;

int64_t ResolveCacheIndex(const std::optional<torch::Tensor>& conv_state_indices,
                          int64_t batch_idx,
                          int64_t slot_idx) {
  if (!conv_state_indices.has_value()) {
    return batch_idx;
  }
  if (conv_state_indices->dim() == 1) {
    return conv_state_indices->index({batch_idx}).item<int32_t>();
  }
  return conv_state_indices->index({batch_idx, slot_idx}).item<int32_t>();
}

int64_t ResolveCacheIndex(const torch::Tensor& conv_state_indices,
                          int64_t batch_idx,
                          int64_t slot_idx) {
  if (conv_state_indices.dim() == 1) {
    return conv_state_indices.index({batch_idx}).item<int32_t>();
  }
  return conv_state_indices.index({batch_idx, slot_idx}).item<int32_t>();
}

torch::Tensor causal_conv1d_update_ref_v2(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    bool activation = true,
    const std::optional<torch::Tensor>& conv_state_indices = std::nullopt,
    const std::optional<torch::Tensor>& query_start_loc = std::nullopt,
    int32_t max_query_len = -1,
    int32_t pad_slot_id = -1,
    const std::optional<torch::Tensor>& block_idx_last_scheduled_token =
        std::nullopt,
    const std::optional<torch::Tensor>& initial_state_idx = std::nullopt) {
  auto dtype_in = x.scalar_type();
  torch::Tensor x_kernel;
  torch::Tensor qsl;
  int64_t batch = 0;
  int64_t dim = 0;
  int64_t seqlen = 0;
  const int64_t width = weight.size(1);
  const int64_t eff_state_len = width - 1;

  if (query_start_loc.has_value()) {
    batch = query_start_loc->size(0) - 1;
    dim = x.size(1);
    seqlen = max_query_len;
    x_kernel = x.contiguous();
    qsl = query_start_loc->to(torch::kInt32).contiguous();
  } else {
    auto x_work = x.dim() == 2 ? x.unsqueeze(-1) : x;
    batch = x_work.size(0);
    dim = x_work.size(1);
    seqlen = x_work.size(2);
    x_kernel = x_work.transpose(1, 2).contiguous().view({batch * seqlen, dim});
    qsl = torch::arange(
        0, (batch + 1) * seqlen, seqlen, torch::TensorOptions().dtype(torch::kInt32));
  }

  torch::Tensor slot_table;
  if (!conv_state_indices.has_value()) {
    auto base =
        torch::arange(batch, torch::TensorOptions().dtype(torch::kInt32));
    slot_table = torch::stack({base, base}, 1).contiguous();
  } else if (conv_state_indices->dim() == 1) {
    auto indices = conv_state_indices->to(torch::kInt32).contiguous();
    slot_table = torch::stack({indices, indices}, 1).contiguous();
  } else {
    slot_table = conv_state_indices->to(torch::kInt32).contiguous();
  }

  auto bias_tensor = bias.has_value()
                         ? bias->to(torch::kFloat).contiguous()
                         : torch::zeros({dim}, torch::TensorOptions().dtype(torch::kFloat));
  auto x_float = x_kernel.to(torch::kFloat).contiguous();
  auto weight_float = weight.to(torch::kFloat).contiguous();
  auto state_float = conv_state.to(torch::kFloat).contiguous().clone();
  auto out_float = torch::empty_like(x_float);

  auto zero_i32 = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32));
  auto last_idx = block_idx_last_scheduled_token.has_value()
                      ? block_idx_last_scheduled_token->to(torch::kInt32).contiguous()
                      : zero_i32;
  auto init_idx = initial_state_idx.has_value()
                      ? initial_state_idx->to(torch::kInt32).contiguous()
                      : zero_i32;

  const auto stride_x_token = x_float.stride(0);
  const auto stride_state_seq = state_float.stride(0);
  const auto stride_state_dim = state_float.stride(1);
  const auto stride_state_token = state_float.stride(2);
  const int64_t num_cache_lines = conv_state.size(0);

  auto* x_ptr = x_float.data_ptr<float>();
  auto* w_ptr = weight_float.data_ptr<float>();
  auto* bias_ptr = bias_tensor.data_ptr<float>();
  auto* state_ptr = state_float.data_ptr<float>();
  auto* out_ptr = out_float.data_ptr<float>();
  auto* qsl_ptr = qsl.data_ptr<int32_t>();

  for (int64_t b = 0; b < batch; ++b) {
    const int64_t start = qsl_ptr[b];
    const int64_t end = qsl_ptr[b + 1];
    const int64_t seq_len = end - start;
    if (seq_len <= 0) {
      continue;
    }

    const int64_t state_len_run = eff_state_len - (seqlen - seq_len);
    const int64_t init_slot = init_idx.index({b}).item<int32_t>();
    const int64_t last_slot = last_idx.index({b}).item<int32_t>();
    const int64_t in_idx = ResolveCacheIndex(slot_table, b, init_slot);
    if (in_idx == pad_slot_id) {
      continue;
    }
    const int64_t out_idx = ResolveCacheIndex(slot_table, b, last_slot);
    const bool input_valid = in_idx >= 0 && in_idx < num_cache_lines;
    const bool output_valid = out_idx >= 0 && out_idx < num_cache_lines;
    const int64_t keep_shift = seq_len < state_len_run ? state_len_run - seq_len : 0;
    const int64_t tail_start = seq_len >= state_len_run ? seq_len - state_len_run : 0;

    for (int64_t feat = 0; feat < dim; ++feat) {
      std::vector<float> cols(std::max<int64_t>(width - 1, 0), 0.0f);
      if (input_valid) {
        for (int64_t k = 0; k < width - 1; ++k) {
          const int64_t offset = in_idx * stride_state_seq +
                                 feat * stride_state_dim +
                                 k * stride_state_token;
          cols[k] = state_ptr[offset];
        }
      }

      for (int64_t t = 0; t < seq_len; ++t) {
        const int64_t x_offset = (start + t) * stride_x_token + feat;
        const float x_val = x_ptr[x_offset];
        float acc = bias_ptr[feat];
        for (int64_t k = 0; k < width - 1; ++k) {
          acc += cols[k] * w_ptr[feat * width + k];
        }
        acc += x_val * w_ptr[feat * width + (width - 1)];
        out_ptr[x_offset] = acc;
        for (int64_t k = 0; k < width - 2; ++k) {
          cols[k] = cols[k + 1];
        }
        if (width > 1) {
          cols[width - 2] = x_val;
        }
      }

      if (!output_valid || state_len_run <= 0) {
        continue;
      }
      if (seq_len < state_len_run) {
        if (input_valid) {
          for (int64_t dst_tok = 0; dst_tok < keep_shift; ++dst_tok) {
            const int64_t src_offset =
                in_idx * stride_state_seq + feat * stride_state_dim +
                (dst_tok + seq_len) * stride_state_token;
            const int64_t dst_offset =
                out_idx * stride_state_seq + feat * stride_state_dim +
                dst_tok * stride_state_token;
            state_ptr[dst_offset] = state_ptr[src_offset];
          }
        }
        for (int64_t x_tok = 0; x_tok < seq_len; ++x_tok) {
          const int64_t dst_tok = keep_shift + x_tok;
          if (dst_tok >= state_len_run) {
            continue;
          }
          const int64_t x_offset = (start + x_tok) * stride_x_token + feat;
          const int64_t dst_offset =
              out_idx * stride_state_seq + feat * stride_state_dim +
              dst_tok * stride_state_token;
          state_ptr[dst_offset] = x_ptr[x_offset];
        }
      } else {
        for (int64_t dst_tok = 0; dst_tok < state_len_run; ++dst_tok) {
          const int64_t x_offset =
              (start + tail_start + dst_tok) * stride_x_token + feat;
          const int64_t dst_offset =
              out_idx * stride_state_seq + feat * stride_state_dim +
              dst_tok * stride_state_token;
          state_ptr[dst_offset] = x_ptr[x_offset];
        }
      }
    }
  }

  auto out = activation ? torch::silu(out_float) : out_float;
  conv_state.copy_(state_float.view_as(conv_state).to(conv_state.scalar_type()));
  return out.to(dtype_in);
}

class TritonCausalConv1dUpdateV2Test : public ::testing::Test {
 protected:
  void SetUp() override {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device("npu:0");
      npu_available_ = true;
    } catch (...) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
      npu_available_ = false;
      return;
    }

    torch::manual_seed(42);
    if (!g_npu_initialized_v2) {
      torch_npu::init_npu(device_str_);
      g_npu_initialized_v2 = true;
    }
    kernel_name_ = "_causal_conv1d_update_kernel_npu_tiled_v2";
    binary_filename_ = "_causal_conv1d_update_kernel_npu_tiled_v2.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& reg = KernelRegistry::get_instance();
    (void)reg.register_kernel(kernel_name_, binary_path_);
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceIdV2);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

namespace {

void RunVarlenApcDecodeCase(const torch::TensorOptions& tensor_options,
                            const std::string& device_str,
                            int64_t dim) {
  auto device = at::Device(device_str);
  constexpr int64_t batch = 4;
  constexpr int64_t width = 4;

  auto x = torch::randn({batch, dim}, tensor_options);
  auto x_ref = x.cpu();

  auto conv_state =
      torch::randn({batch * 2, dim, width - 1}, tensor_options);
  auto conv_state_before = conv_state.detach().clone().cpu();
  auto conv_state_ref = conv_state.detach().clone().cpu();

  auto weight = torch::randn({dim, width}, tensor_options);
  auto weight_ref = weight.cpu();
  auto bias = torch::zeros({dim}, tensor_options);
  auto bias_ref = bias.cpu();

  auto base_idx =
      torch::arange(batch, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto conv_state_indices = torch::stack({base_idx, base_idx + batch}, 1).contiguous();
  auto block_idx_last_scheduled_token =
      torch::ones({batch}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto initial_state_idx =
      torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto query_start_loc =
      torch::arange(0,
                    batch + 1,
                    torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto out_ref = causal_conv1d_update_ref_v2(x_ref,
                                             conv_state_ref,
                                             weight_ref,
                                             bias_ref,
                                             true,
                                             conv_state_indices.cpu(),
                                             query_start_loc.cpu(),
                                             1,
                                             -1,
                                             block_idx_last_scheduled_token.cpu(),
                                             initial_state_idx.cpu());

  auto out = npu_causal_conv1d_update_v2(x,
                                         conv_state,
                                         weight,
                                         true,
                                         bias,
                                         conv_state_indices,
                                         query_start_loc,
                                         1,
                                         -1,
                                         block_idx_last_scheduled_token,
                                         initial_state_idx,
                                         false);
  aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(kDeviceIdV2).stream());

  auto out_cpu = out.cpu();
  EXPECT_TRUE(torch::allclose(out_cpu, out_ref, 1e-2, kToleranceV2));

  auto conv_state_cpu = conv_state.cpu();
  EXPECT_TRUE(torch::allclose(conv_state_cpu.slice(0, 0, batch),
                              conv_state_before.slice(0, 0, batch),
                              1e-2,
                              kToleranceV2));
  EXPECT_FALSE(torch::allclose(conv_state_cpu.slice(0, batch, batch * 2),
                               conv_state_before.slice(0, batch, batch * 2),
                               1e-2,
                               kToleranceV2));
}

std::vector<int64_t> QwenGdnMixedQkvDims() {
  return {
      768,    // 16->16, TP8 ratio 1
      1024,   // 16->32 or 16->48, TP8 ratio 2
      1280,   // 16->48, TP8 ratio 3
      1536,   // 16->64, TP8 ratio 4
      2048,   // 16->32, TP4 ratio 2
      2560,   // 16->48, TP4 ratio 3
      3072,   // 16->64, TP4 ratio 4
      4096,   // 16->32, TP2 ratio 2
      5120,   // 16->48, TP2 ratio 3
      6144,   // 16->64, TP2 ratio 4
      8192,   // 16->32, TP1 ratio 2
      10240,  // 16->48, TP1 ratio 3
      12288,  // 16->64, TP1 ratio 4
  };
}

void RunDenseNoBiasNoActivationCase(const torch::TensorOptions& tensor_options,
                                    int64_t dim) {
  constexpr int64_t batch = 2;
  constexpr int64_t width = 2;
  constexpr int64_t seqlen = 3;

  auto x = torch::randn({batch, dim, seqlen}, tensor_options);
  auto x_ref = x.cpu();
  auto conv_state = torch::randn({batch, dim, width - 1}, tensor_options);
  auto conv_state_ref = conv_state.detach().clone().cpu();
  auto weight = torch::randn({dim, width}, tensor_options);
  auto weight_ref = weight.cpu();

  auto out_ref = causal_conv1d_update_ref_v2(
      x_ref, conv_state_ref, weight_ref, std::nullopt, false);

  auto out = npu_causal_conv1d_update_v2(x,
                                         conv_state,
                                         weight,
                                         false,
                                         std::nullopt,
                                         std::nullopt,
                                         std::nullopt,
                                         -1,
                                         -1,
                                         std::nullopt,
                                         std::nullopt,
                                         false);
  aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(kDeviceIdV2).stream());

  EXPECT_TRUE(torch::allclose(out.cpu(), out_ref, 1e-2, kToleranceV2));
  EXPECT_TRUE(
      torch::allclose(conv_state.cpu(), conv_state_ref, 1e-2, kToleranceV2));
}

void RunVarlenPadWidth5Case(const torch::TensorOptions& tensor_options,
                            const std::string& device_str) {
  auto device = at::Device(device_str);
  constexpr int64_t dim = 1024;
  constexpr int64_t width = 5;
  constexpr int64_t max_query_len = 3;

  auto x = torch::randn({5, dim}, tensor_options);
  auto x_ref = x.cpu();
  auto conv_state = torch::randn({4, dim, width - 1}, tensor_options);
  auto conv_state_before = conv_state.detach().clone().cpu();
  auto conv_state_ref = conv_state.detach().clone().cpu();
  auto weight = torch::randn({dim, width}, tensor_options);
  auto weight_ref = weight.cpu();
  auto bias = torch::zeros({dim}, tensor_options);
  auto bias_ref = bias.cpu();
  auto conv_state_indices = torch::tensor(
      {{0, 1}, {-7, -7}, {2, 3}},
      torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto block_idx_last_scheduled_token = torch::tensor(
      {1, 0, 1}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto initial_state_idx = torch::zeros(
      {3}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  auto query_start_loc = torch::tensor(
      {0, 3, 3, 5}, torch::TensorOptions().dtype(torch::kInt32).device(device));

  auto out_ref = causal_conv1d_update_ref_v2(x_ref,
                                             conv_state_ref,
                                             weight_ref,
                                             bias_ref,
                                             true,
                                             conv_state_indices.cpu(),
                                             query_start_loc.cpu(),
                                             max_query_len,
                                             -7,
                                             block_idx_last_scheduled_token.cpu(),
                                             initial_state_idx.cpu());

  auto out = npu_causal_conv1d_update_v2(x,
                                         conv_state,
                                         weight,
                                         true,
                                         bias,
                                         conv_state_indices,
                                         query_start_loc,
                                         max_query_len,
                                         -7,
                                         block_idx_last_scheduled_token,
                                         initial_state_idx,
                                         false);
  aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(kDeviceIdV2).stream());

  EXPECT_TRUE(torch::allclose(out.cpu(), out_ref, 1e-2, kToleranceV2));
  auto conv_state_cpu = conv_state.cpu();
  EXPECT_TRUE(torch::allclose(conv_state_cpu.index({0}),
                              conv_state_before.index({0}),
                              1e-2,
                              kToleranceV2));
  EXPECT_TRUE(torch::allclose(conv_state_cpu.index({2}),
                              conv_state_before.index({2}),
                              1e-2,
                              kToleranceV2));
  EXPECT_FALSE(torch::allclose(conv_state_cpu.index({1}),
                               conv_state_before.index({1}),
                               1e-2,
                               kToleranceV2));
  EXPECT_FALSE(torch::allclose(conv_state_cpu.index({3}),
                               conv_state_before.index({3}),
                               1e-2,
                               kToleranceV2));
}

}  // namespace

TEST_F(TritonCausalConv1dUpdateV2Test, VarlenApcDecodeTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  RunVarlenApcDecodeCase(tensor_options_, device_str_, 2048);
}

TEST_F(TritonCausalConv1dUpdateV2Test, VarlenApcDecodeDim1024Test) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  RunVarlenApcDecodeCase(tensor_options_, device_str_, 1024);
}

TEST_F(TritonCausalConv1dUpdateV2Test, QwenGdnMixedQkvDimsTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  for (const int64_t dim : QwenGdnMixedQkvDims()) {
    SCOPED_TRACE("dim=" + std::to_string(dim));
    RunVarlenApcDecodeCase(tensor_options_, device_str_, dim);
  }
}

TEST_F(TritonCausalConv1dUpdateV2Test, DenseNoBiasNoActivationWidth2Test) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  RunDenseNoBiasNoActivationCase(tensor_options_, 1024);
}

TEST_F(TritonCausalConv1dUpdateV2Test, VarlenPadWidth5Test) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  RunVarlenPadWidth5Case(tensor_options_, device_str_);
}

}  // namespace xllm::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  xllm::testing::init_npu_test_runtime();
  return RUN_ALL_TESTS();
}
