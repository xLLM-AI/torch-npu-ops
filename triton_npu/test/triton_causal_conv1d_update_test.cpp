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
#include <torch/nn/functional/conv.h>
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

constexpr float kTolerance = 5e-2f;
constexpr int32_t kDeviceId = 0;

#include <torch/torch.h>

#include <string>
#include <vector>

torch::Tensor causal_conv1d_update_ref(
    const torch::Tensor& x,
    torch::Tensor& conv_state,
    const torch::Tensor& weight,
    const std::optional<torch::Tensor>& bias = std::nullopt,
    bool activation = true,
    const std::optional<torch::Tensor>& cache_seqlens = std::nullopt) {
  auto dtype_in = x.scalar_type();
  bool unsqueeze = (x.dim() == 2);
  torch::Tensor x_ = unsqueeze ? x.unsqueeze(-1) : x;
  int64_t batch = x_.size(0);
  int64_t dim = x_.size(1);
  int64_t seqlen = x_.size(2);
  int64_t width = weight.size(1);
  int64_t state_len = conv_state.size(2);

  TORCH_CHECK(
      conv_state.sizes().vec() == std::vector<int64_t>({batch, dim, state_len}),
      "conv_state shape mismatch. Expected: (",
      batch,
      ", ",
      dim,
      ", ",
      state_len,
      "), Got: ",
      conv_state.sizes());
  TORCH_CHECK(weight.sizes().vec() == std::vector<int64_t>({dim, width}),
              "weight shape mismatch. Expected: (",
              dim,
              ", ",
              width,
              "), Got: ",
              weight.sizes());

  torch::Tensor x_new;
  if (!cache_seqlens.has_value()) {
    x_new = torch::cat({conv_state, x_}, -1).to(weight.scalar_type());
    conv_state.copy_(x_new.slice(-1, -state_len, x_new.size(-1)));
  } else {
    auto cache_seqlens_tensor = cache_seqlens.value();
    TORCH_CHECK(
        cache_seqlens_tensor.sizes().vec() == std::vector<int64_t>({batch}),
        "cache_seqlens must be 1D tensor of size (batch,)");

    cache_seqlens_tensor = cache_seqlens_tensor.to(x_.device());
    auto arange_tensor =
        torch::arange(-(width - 1), 0, torch::kLong).to(x_.device());
    auto width_idx = arange_tensor.unsqueeze(0)
                         .add(cache_seqlens_tensor.unsqueeze(1))
                         .to(torch::kLong);

    width_idx = torch::remainder(width_idx, state_len)
                    .unsqueeze(1)
                    .expand({batch, dim, width - 1});

    auto state_gathered = conv_state.gather(2, width_idx);
    x_new = torch::cat({state_gathered, x_}, -1).to(weight.scalar_type());

    auto copy_idx = torch::arange(0, seqlen, torch::kLong)
                        .to(x_.device())
                        .unsqueeze(0)
                        .add(cache_seqlens_tensor.unsqueeze(1));
    copy_idx = torch::remainder(copy_idx, state_len)
                   .unsqueeze(1)
                   .expand({batch, dim, seqlen})
                   .to(torch::kLong);
    conv_state.scatter_(2, copy_idx, x_);
  }

  torch::Tensor bias_tensor = bias.has_value() ? bias.value() : torch::Tensor();
  auto weight_4d = weight.unsqueeze(1);
  torch::Tensor out = torch::conv1d(x_new,
                                    weight_4d,
                                    bias_tensor,
                                    /*stride=*/torch::IntArrayRef{1},
                                    /*padding=*/torch::IntArrayRef{0},
                                    /*dilation=*/torch::IntArrayRef{1},
                                    /*groups=*/static_cast<int64_t>(dim));
  out = out.slice(-1, -seqlen, out.size(-1));

  if (activation) {
    out = torch::silu(out);
  }
  if (unsqueeze) {
    out = out.squeeze(-1);
  }
  return out.to(dtype_in);
}

class TritonCausalConv1dUpdateTest : public ::testing::Test {
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
    c10_npu::SetDevice(kDeviceId);
    kernel_name_ = "_causal_conv1d_update_qwen_decode_kernel";
    binary_filename_ = "_causal_conv1d_update_qwen_decode_kernel.npubin";
    binary_path_ = GetKernelBinaryPath(binary_filename_);
    auto& reg = KernelRegistry::get_instance();
    (void)reg.register_kernel(kernel_name_, binary_path_);
  }

  void TearDown() override {
    if (npu_available_) {
      try {
      } catch (...) {
      }
    }
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
  std::string binary_filename_;
  std::string kernel_name_;
  std::string binary_path_;
};

TEST_F(TritonCausalConv1dUpdateTest, MultiBatchTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);
  constexpr int64_t batch = 4;
  constexpr int64_t width = 4;
  constexpr int64_t seqlen = 1;
  constexpr bool has_bias = false;
  constexpr bool silu_activation = true;

  torch::manual_seed(0);
  auto dtype = torch::kBFloat16;
  float atol = 5e-2f;

  for (int64_t dim : {2048, 5120}) {
    auto x = torch::randn({batch, dim, seqlen},
                          torch::TensorOptions().dtype(dtype).device(device));
    auto x_ref = x.clone().cpu();

    auto conv_state =
        torch::randn({batch, dim, width - 1},
                     torch::TensorOptions().dtype(dtype).device(device));
    auto conv_state_ref = conv_state.detach().clone().cpu();

    auto weight = torch::randn(
        {dim, width}, torch::TensorOptions().dtype(dtype).device(device));
    auto weight_ref = weight.clone().cpu();

    std::optional<torch::Tensor> bias = std::nullopt;
    if (has_bias) {
      bias = torch::randn({dim},
                          torch::TensorOptions().dtype(dtype).device(device));
    }

    auto conv_state_indices = torch::arange(
        batch, torch::TensorOptions().dtype(torch::kInt32).device(device));

    auto out_ref = causal_conv1d_update_ref(x_ref,
                                            conv_state_ref,
                                            weight_ref,
                                            std::nullopt,
                                            silu_activation,
                                            std::nullopt);

    auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
    auto out =
        npu_causal_conv1d_update(x,
                                 conv_state,
                                 weight,
                                 silu_activation,
                                 bias,
                                 std::nullopt,
                                 conv_state_indices,
                                 std::nullopt,
                                 std::nullopt,
                                 -1,
                                 std::nullopt,
                                 -1,
                                 false);
    aclrtSynchronizeStream(npu_stream.stream());

    auto out_cpu = out.cpu();
    auto output_diff = (out_ref - out_cpu).abs();
    float max_diff = output_diff.max().item().toFloat();
    float dim_atol = dim > 2048 ? atol * 2.0f : atol;

    EXPECT_LT(max_diff, dim_atol)
        << "Output mismatch: max diff = " << max_diff
        << ", tolerance = " << dim_atol << ", dim = " << dim;

    auto conv_state_cpu = conv_state.cpu();
    auto state_diff = (conv_state_ref - conv_state_cpu).abs();
    float max_state_diff = state_diff.max().item().toFloat();
    EXPECT_LT(max_state_diff, atol * 2.0f)
        << "Conv state mismatch: max diff = " << max_state_diff
        << ", tolerance = " << (atol * 2.0f) << ", dim = " << dim;
  }
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
