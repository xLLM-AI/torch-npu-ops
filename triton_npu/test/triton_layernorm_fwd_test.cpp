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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>
#include <torch_npu/csrc/core/npu/NPUFunctions.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <vector>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {
constexpr int32_t kDeviceId = 0;
constexpr float kTolerance = 1e-3f;
constexpr float kBfloat16Tolerance = 5e-2f;

torch::Tensor layer_norm_golden_cpu(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    double eps = 1e-6,
    const std::optional<torch::Tensor>& z = std::nullopt,
    const int64_t group_size = -1,
    bool norm_before_gate = true,
    bool is_rms_norm = false) {
  auto x_shape_og = x.sizes();
  int64_t last_dim = x.size(-1);
  torch::Tensor x_2d = x.reshape({-1, last_dim})
                           .cpu()
                           .contiguous();  // (M, N) = (batch*seq_len, feat_dim)
  int64_t M = x_2d.size(0);
  int64_t N = x_2d.size(1);

  torch::Tensor weight_cpu = weight.cpu().contiguous();
  torch::Tensor bias_cpu;
  if (bias.defined()) {
    bias_cpu = bias.cpu().contiguous();
  }

  torch::Tensor z_2d;
  if (z.has_value()) {
    TORCH_CHECK(z->stride(-1) == 1, "z stride(-1) must be 1");
    TORCH_CHECK(z->sizes() == x.sizes(), "z shape must match x");
    z_2d = z->cpu().reshape({-1, last_dim}).contiguous();
  }

  int64_t group_size_val = group_size;

  TORCH_CHECK(N % group_size_val == 0,
              "N=" + std::to_string(N) + " must be divisible by group_size=" +
                  std::to_string(group_size_val));
  int64_t ngroups = N / group_size_val;

  torch::Tensor x_input;
  if (z_2d.defined() && !norm_before_gate) {
    torch::Tensor gate = z_2d * torch::sigmoid(z_2d);
    x_input = x_2d * gate;
  } else {
    x_input = x_2d;
  }

  torch::Tensor x_grouped =
      x_input.unfold(1, group_size, group_size);  // (M, ngroups, group_size)
  torch::Tensor x_grouped_flat =
      x_grouped.reshape({-1, group_size});  // (M*ngroups, group_size)

  torch::Tensor x_norm_flat;
  if (!is_rms_norm) {
    x_norm_flat =
        torch::layer_norm(x_grouped_flat,
                          torch::IntArrayRef({group_size}),  // normalized_shape
                          torch::Tensor(),                   // weight = None
                          torch::Tensor(),                   // bias = None
                          eps);
  } else {
    torch::Tensor x_sq = torch::pow(x_grouped_flat, 2);
    torch::Tensor mean_sq =
        torch::mean(x_sq, /*dim=*/-1, /*keepdim=*/true);  // (M*ngroups, 1)
    torch::Tensor rsqrt_mean_sq =
        torch::rsqrt(mean_sq + eps);  // 1 / sqrt(mean_sq + eps)
    x_norm_flat = x_grouped_flat * rsqrt_mean_sq;
  }

  torch::Tensor x_norm_grouped = x_norm_flat.reshape({M, ngroups, group_size});
  torch::Tensor x_norm = x_norm_grouped.contiguous().view({M, N});
  torch::Tensor y = x_norm * weight_cpu;
  if (bias_cpu.defined()) {
    y = y + bias_cpu;
  }
  if (z_2d.defined() && norm_before_gate) {
    torch::Tensor gate = z_2d * torch::sigmoid(z_2d);  // (M, N)
    y = y * gate;
  }

  return y.reshape(x_shape_og);
}

class TritonLayerNormFwdTest : public ::testing::Test {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      c10_npu::SetDevice(kDeviceId);
      npu_initialized_ = true;
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
      return;
    }
    torch::manual_seed(42);
  }

  void TearDown() override {}

  bool npu_available_ = false;
  std::string device_str_ = "npu:" + std::to_string(kDeviceId);
};

bool TritonLayerNormFwdTest::npu_initialized_ = false;

// (2, 8, 128, False, True, True, false, None),
// x shape:[2*8, 128]
// weight shape:[128,]
// z shape:[2*8, 128]
// group_size:None
// norm_before_gate = true
// bool is_rms_norm = false
TEST_F(TritonLayerNormFwdTest, KernelTestFast) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);

  // Shape targeting fast kernel: group_size=128, no z, not rms_norm
  int64_t batch_size = 32768;
  int64_t seq_len = 1;
  int64_t hidden_dim = 128;
  float eps = 1e-6;
  int64_t group_size = hidden_dim;

  auto tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto x = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
  auto weight = torch::randn({hidden_dim}, tensor_options);
  auto bias = torch::randn({hidden_dim}, tensor_options);
  std::optional<torch::Tensor> z_optional;

  auto output_golden = layer_norm_golden_cpu(
      x, weight, bias, eps, z_optional, group_size, true, false);
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto output = xllm::kernel::npu::layer_norm_fwd(
      x, weight, bias, eps, z_optional, group_size, true, false);
  aclrtSynchronizeStream(npu_stream.stream());

  auto output_golden_cpu = output_golden.cpu().contiguous();
  auto output_cpu = output.cpu().contiguous();

  auto output_diff = torch::abs(output_cpu - output_golden_cpu);
  float output_max_diff = torch::max(output_diff).item().to<float>();

  EXPECT_LT(output_max_diff, kTolerance)
      << "Fast LayerNorm output max diff (" << output_max_diff
      << ") > tolerance (" << kTolerance << ")";
}

TEST_F(TritonLayerNormFwdTest, KernelTest2) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);

  int64_t batch_size = 2;
  int64_t seq_len = 8;
  int64_t hidden_dim = 128;
  float eps = 1e-6;
  int64_t group_size = hidden_dim;

  auto tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto x = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
  auto weight = torch::randn({hidden_dim}, tensor_options);
  torch::Tensor bias;
  auto z = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
  std::optional<torch::Tensor> z_optional = z;

  auto output_golden = layer_norm_golden_cpu(
      x, weight, bias, eps, z_optional, group_size, true, true);
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto output = xllm::kernel::npu::layer_norm_fwd(
      x, weight, bias, eps, z_optional, group_size, true, true);
  aclrtSynchronizeStream(npu_stream.stream());

  auto output_golden_cpu = output_golden.cpu().contiguous();
  auto output_cpu = output.cpu().contiguous();

  auto output_diff = torch::abs(output_cpu - output_golden_cpu);
  float output_max_diff = torch::max(output_diff).item().to<float>();

  EXPECT_LT(output_max_diff, kTolerance)
      << "LayerNorm output max diff (" << output_max_diff << ") > tolerance ("
      << kTolerance << ")";
}

// ---------------------------------------------------------------------------
// Performance comparison: same (32768, 128) shape on both fast & fallback
// ---------------------------------------------------------------------------
TEST_F(TritonLayerNormFwdTest, KernelTestFallbackLarge) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  auto device = at::Device(device_str_);

  int64_t batch_size = 32768;
  int64_t seq_len = 1;
  int64_t hidden_dim = 128;
  float eps = 1e-6;
  int64_t group_size = hidden_dim;

  auto tensor_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);

  auto x = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
  auto weight = torch::randn({hidden_dim}, tensor_options);
  torch::Tensor bias;
  auto z = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options);
  std::optional<torch::Tensor> z_optional = z;

  // Force fallback path via is_rms_norm=true (z is provided to match the
  // existing AOT binary signature which was compiled with HAS_Z=True)
  auto output_golden = layer_norm_golden_cpu(
      x, weight, bias, eps, z_optional, group_size, true, true);
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto output = xllm::kernel::npu::layer_norm_fwd(
      x, weight, bias, eps, z_optional, group_size, true, true);
  aclrtSynchronizeStream(npu_stream.stream());

  auto output_golden_cpu = output_golden.cpu().contiguous();
  auto output_cpu = output.cpu().contiguous();

  auto output_diff = torch::abs(output_cpu - output_golden_cpu);
  float output_max_diff = torch::max(output_diff).item().to<float>();

  EXPECT_LT(output_max_diff, kTolerance)
      << "Fallback (large) LayerNorm output max diff (" << output_max_diff
      << ") > tolerance (" << kTolerance << ")";
}

TEST_F(TritonLayerNormFwdTest, KernelTestFastRMSBF16ZNoBiasNoMask) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  ASSERT_TRUE(KernelBinaryExists(
      "layer_norm_fwd_kernel_fast_rms_bf16_z_nobias_nomask.npubin"));

  auto device = at::Device(device_str_);

  int64_t batch_size = 1024;
  int64_t seq_len = 1;
  int64_t hidden_dim = 128;
  float eps = 1e-6;
  int64_t group_size = hidden_dim;
  float scale = 0.5f;

  auto tensor_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  auto x = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options) *
           scale;
  auto weight = torch::randn({hidden_dim}, tensor_options) * scale;
  torch::Tensor bias;
  auto z = torch::randn({batch_size, seq_len, hidden_dim}, tensor_options) *
           scale;
  std::optional<torch::Tensor> z_optional = z;

  auto output_golden = layer_norm_golden_cpu(
      x, weight, bias, eps, z_optional, group_size, true, true);
  auto npu_stream = c10_npu::getCurrentNPUStream(0);
  auto output = xllm::kernel::npu::layer_norm_fwd(
      x, weight, bias, eps, z_optional, group_size, true, true);
  aclrtSynchronizeStream(npu_stream.stream());

  EXPECT_TRUE(KernelRegistry::get_instance().is_kernel_registered(
      "layer_norm_fwd_kernel_fast_rms_bf16_z_nobias_nomask"));

  auto output_golden_cpu = output_golden.cpu().contiguous();
  auto output_cpu = output.cpu().contiguous();

  auto output_diff = torch::abs(output_cpu.to(torch::kFloat32) -
                               output_golden_cpu.to(torch::kFloat32));
  float output_max_diff = torch::max(output_diff).item().to<float>();

  EXPECT_LT(output_max_diff, kBfloat16Tolerance)
      << "No-mask bf16 RMS LayerNorm output max diff (" << output_max_diff
      << ") > tolerance (" << kBfloat16Tolerance << ")";
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
