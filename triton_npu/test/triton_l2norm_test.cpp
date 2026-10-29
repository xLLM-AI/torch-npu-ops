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

#include <tuple>
#include <vector>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"

namespace xllm::kernel::npu {

constexpr int32_t kDeviceId = 0;
constexpr float kTolerance = 5e-2f;

torch::Tensor l2norm_ref(const torch::Tensor& x, float eps = 1e-6f) {
  auto x_fp32 = x.to(torch::kFloat32);
  auto out = x_fp32 * torch::rsqrt(torch::sum(x_fp32 * x_fp32, -1, true) + eps);
  return out.to(x.scalar_type());
}

class TritonL2NormTest : public ::testing::TestWithParam<std::vector<int64_t>> {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      c10_npu::SetDevice(kDeviceId);
      auto& reg = KernelRegistry::get_instance();
      std::string binary_path =
          GetKernelBinaryPath("l2norm_fwd_kernel2_loop.npubin");
      npu_initialized_ =
          reg.register_kernel("l2norm_fwd_kernel2_loop", binary_path) &&
          reg.get_kernel_stub("l2norm_fwd_kernel2_loop") != nullptr;
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
    } catch (...) {
    }
  }

  void SetUp() override {
    npu_available_ = npu_initialized_;
    if (!npu_available_) {
      tensor_options_ =
          torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
      return;
    }
    tensor_options_ = torch::TensorOptions()
                          .dtype(torch::kBFloat16)
                          .device("npu:" + std::to_string(kDeviceId));
    device_str_ = "npu:" + std::to_string(kDeviceId);
    torch::manual_seed(42);
  }

  torch::TensorOptions tensor_options_;
  bool npu_available_ = false;
  std::string device_str_;
};

bool TritonL2NormTest::npu_initialized_ = false;

TEST_P(TritonL2NormTest, DynamicLastDimAgainstCpuReference) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const auto shape = GetParam();
  auto device = at::Device(device_str_);
  auto options_cpu =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);
  auto options_npu =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);

  auto x_cpu = torch::randn(shape, options_cpu);
  auto x_npu = x_cpu.to(device).contiguous();

  auto ref = l2norm_ref(x_cpu);
  auto out = npu_l2norm_last_dim(x_npu);

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  aclrtSynchronizeStream(npu_stream.stream());

  auto out_cpu = out.cpu();
  auto diff = torch::abs(out_cpu.to(torch::kFloat32) - ref.to(torch::kFloat32));
  float max_diff = torch::max(diff).item<float>();

  EXPECT_LT(max_diff, kTolerance)
      << "l2norm output max diff (" << max_diff << ") > tolerance ("
      << kTolerance << ")";
}

INSTANTIATE_TEST_SUITE_P(
    L2NormCases,
    TritonL2NormTest,
    ::testing::Values(std::vector<int64_t>{1, 31, 4, 64},
                      std::vector<int64_t>{1, 31, 4, 128},
                      std::vector<int64_t>{2, 17, 3, 192},
                      std::vector<int64_t>{2, 64, 8, 256}));

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
