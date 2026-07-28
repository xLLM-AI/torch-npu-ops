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
#include <cstdlib>
#include <optional>
#include <string>
#include <tuple>

#include "kernel_registry.h"
#include "test/test_utils.h"
#include "tests/npu_test_environment.h"
#include "torch_api/triton_ops_api.h"
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {
namespace {

constexpr int32_t kDeviceId = 0;
constexpr double kDefaultEps = 1e-6;
constexpr float kTolerance = 2e-1f;
constexpr int64_t kDefaultNumTokens = 10;

std::optional<std::string> GetEnv(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  return std::string(value);
}

int64_t GetEnvInt(const char* name, int64_t default_value) {
  auto value = GetEnv(name);
  if (!value.has_value()) {
    return default_value;
  }
  return std::stoll(value.value());
}

double GetEnvDouble(const char* name, double default_value) {
  auto value = GetEnv(name);
  if (!value.has_value()) {
    return default_value;
  }
  return std::stod(value.value());
}

std::string FormatKernelEps(double eps) {
  std::string eps_str = std::to_string(eps);
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), '.'),
                eps_str.end());
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), 'e'),
                eps_str.end());
  eps_str.erase(std::remove(eps_str.begin(), eps_str.end(), '-'),
                eps_str.end());
  return eps_str;
}

std::string KernelName(int64_t q_hidden_size,
                       int64_t kv_hidden_size,
                       int64_t head_dim,
                       double eps,
                       bool bias) {
  return "split_rmsnorm_rope_kernel_bias" + std::string(bias ? "1" : "0") +
         "_eps" + FormatKernelEps(eps) + "_hd" + std::to_string(head_dim) +
         "_qh" + std::to_string(q_hidden_size) + "_kvh" +
         std::to_string(kv_hidden_size);
}

torch::Tensor RmsNorm(const torch::Tensor& x,
                      const torch::Tensor& weight,
                      double eps) {
  auto x_f = x.to(torch::kFloat32);
  auto w_f = weight.to(torch::kFloat32);
  auto var = x_f.pow(2).mean(-1, true);
  return x_f * torch::rsqrt(var + eps) * w_f;
}

torch::Tensor RotateHalf(const torch::Tensor& x) {
  const int64_t half = x.size(-1) / 2;
  auto x1 = x.slice(-1, 0, half);
  auto x2 = x.slice(-1, half);
  return torch::cat({-x2, x1}, -1);
}

torch::Tensor ApplyRope(const torch::Tensor& x,
                        const torch::Tensor& sin,
                        const torch::Tensor& cos) {
  auto sin_exp = sin.to(torch::kFloat32).unsqueeze(1);
  auto cos_exp = cos.to(torch::kFloat32).unsqueeze(1);
  auto x_f = x.to(torch::kFloat32);
  return x_f * cos_exp + RotateHalf(x_f) * sin_exp;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Ref(
    const torch::Tensor& qkv,
    const torch::Tensor& sin,
    const torch::Tensor& cos,
    const torch::Tensor& q_weight,
    const torch::Tensor& k_weight,
    int64_t q_hidden_size,
    int64_t kv_hidden_size,
    int64_t head_dim,
    double eps) {
  const int64_t num_tokens = qkv.size(0);
  auto q = qkv.slice(1, 0, q_hidden_size);
  auto k = qkv.slice(1, q_hidden_size, q_hidden_size + kv_hidden_size);
  auto v = qkv.slice(1, q_hidden_size + kv_hidden_size,
                     q_hidden_size + 2 * kv_hidden_size);

  q = q.reshape({num_tokens, q_hidden_size / head_dim, head_dim});
  k = k.reshape({num_tokens, kv_hidden_size / head_dim, head_dim});
  q = RmsNorm(q, q_weight, eps);
  k = RmsNorm(k, k_weight, eps);
  q = ApplyRope(q, sin, cos).reshape({num_tokens, q_hidden_size});
  k = ApplyRope(k, sin, cos).reshape({num_tokens, kv_hidden_size});
  return std::make_tuple(q, k, v.to(torch::kFloat32));
}

}  // namespace

class TritonSplitRmsnormRopeTest : public ::testing::Test {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      torch_npu::init_npu("npu:" + std::to_string(kDeviceId));
      npu_initialized_ = true;
    } catch (...) {
      npu_initialized_ = false;
    }
  }

  static void TearDownTestSuite() {
    if (npu_initialized_) {
      try {
        KernelRegistry::get_instance().cleanup();
        torch_npu::finalize_npu();
      } catch (...) {
      }
    }
  }
};

bool TritonSplitRmsnormRopeTest::npu_initialized_ = false;

TEST_F(TritonSplitRmsnormRopeTest, RandomInputMatchesCpuReference) {
  if (!npu_initialized_) {
    GTEST_SKIP() << "NPU device not available";
  }

  torch::manual_seed(42);

  const int64_t num_tokens =
      GetEnvInt("SPLIT_RMSNORM_ROPE_NUM_TOKENS", kDefaultNumTokens);
  const int64_t head_dim = GetEnvInt("SPLIT_RMSNORM_ROPE_HEAD_DIM", 128);
  const int64_t q_hidden_size =
      GetEnvInt("SPLIT_RMSNORM_ROPE_Q_HIDDEN_SIZE", 8192);
  const int64_t kv_hidden_size =
      GetEnvInt("SPLIT_RMSNORM_ROPE_KV_HIDDEN_SIZE", 1024);
  const double eps = GetEnvDouble("SPLIT_RMSNORM_ROPE_EPS", kDefaultEps);
  const bool bias = false;
  const int64_t total_hidden_size = q_hidden_size + 2 * kv_hidden_size;

  ASSERT_GT(num_tokens, 0);
  ASSERT_GT(head_dim, 0);
  ASSERT_EQ(head_dim % 2, 0);
  ASSERT_EQ(q_hidden_size % head_dim, 0);
  ASSERT_EQ(kv_hidden_size % head_dim, 0);

  auto options_fp32 =
      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
  auto qkv_fp32 = torch::randn({num_tokens, total_hidden_size}, options_fp32);
  auto sin_fp32 = torch::randn({num_tokens, head_dim}, options_fp32);
  auto cos_fp32 = torch::randn({num_tokens, head_dim}, options_fp32);
  auto q_weight_fp32 = torch::randn({head_dim}, options_fp32);
  auto k_weight_fp32 = torch::randn({head_dim}, options_fp32);

  // Match the values actually consumed by the NPU kernel.
  auto qkv = qkv_fp32.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
  auto sin = sin_fp32.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
  auto cos = cos_fp32.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
  auto q_weight =
      q_weight_fp32.to(torch::kBFloat16).to(torch::kFloat32).contiguous();
  auto k_weight =
      k_weight_fp32.to(torch::kBFloat16).to(torch::kFloat32).contiguous();

  const std::string kernel_name =
      KernelName(q_hidden_size, kv_hidden_size, head_dim, eps, bias);
  const std::string binary_filename = kernel_name + ".npubin";
  ASSERT_TRUE(KernelBinaryExists(binary_filename))
      << "missing kernel binary: " << GetKernelBinaryPath(binary_filename);
  auto& reg = KernelRegistry::get_instance();
  ASSERT_TRUE(reg.register_kernel(kernel_name,
                                  GetKernelBinaryPath(binary_filename)));
  ASSERT_NE(reg.get_kernel_stub(kernel_name), nullptr);

  auto [q_ref, k_ref, v_ref] =
      Ref(qkv, sin, cos, q_weight, k_weight, q_hidden_size, kv_hidden_size,
          head_dim, eps);

  auto device = at::Device("npu:" + std::to_string(kDeviceId));
  auto qkv_npu = qkv_fp32.to(torch::kBFloat16).to(device).contiguous();
  auto sin_npu = sin_fp32.to(torch::kBFloat16).to(device).contiguous();
  auto cos_npu = cos_fp32.to(torch::kBFloat16).to(device).contiguous();
  auto q_weight_npu =
      q_weight_fp32.to(torch::kBFloat16).to(device).contiguous();
  auto k_weight_npu =
      k_weight_fp32.to(torch::kBFloat16).to(device).contiguous();

  auto [q_out, k_out, v_out] =
      npu_split_rmsnorm_rope(qkv_npu,
                             sin_npu,
                             cos_npu,
                             q_weight_npu,
                             k_weight_npu,
                             q_hidden_size,
                             kv_hidden_size,
                             head_dim,
                             eps,
                             std::nullopt,
                             std::nullopt,
                             bias);

  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  ASSERT_EQ(aclrtSynchronizeStream(npu_stream.stream()), ACL_ERROR_NONE);

  auto q_cpu = q_out.cpu().contiguous().to(torch::kFloat32);
  auto k_cpu = k_out.cpu().contiguous().to(torch::kFloat32);
  auto v_cpu = v_out.cpu().contiguous().to(torch::kFloat32);

  auto max_q_diff = torch::max(torch::abs(q_cpu - q_ref)).item<float>();
  auto max_k_diff = torch::max(torch::abs(k_cpu - k_ref)).item<float>();
  auto max_v_diff = torch::max(torch::abs(v_cpu - v_ref)).item<float>();

  EXPECT_LT(max_q_diff, kTolerance) << "q max diff";
  EXPECT_LT(max_k_diff, kTolerance) << "k max diff";
  EXPECT_LT(max_v_diff, kTolerance) << "v max diff";
}

}  // namespace xllm::kernel::npu

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  xllm::testing::init_npu_test_runtime();
  return RUN_ALL_TESTS();
}

