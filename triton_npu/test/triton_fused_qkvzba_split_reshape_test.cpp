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
 * C++ unit-test for npu_fused_qkvzba_split_reshape_cat.
 *
 * The test follows the same pattern as triton_rope_inplace_test.cpp:
 *   1. Build CPU golden output with pure-PyTorch/libtorch operations.
 *   2. Run the Triton kernel on NPU.
 *   3. Compare results within bfloat16 tolerance.
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
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace xllm::kernel::npu {

namespace {

constexpr const char* kFusedQkvzbaSplitReshapeKernels[] = {
    "fused_qkvzba_split_reshape_cat_gqa_r1_kernel",
    "fused_qkvzba_split_reshape_cat_gqa_r2_kernel",
    "fused_qkvzba_split_reshape_cat_gqa_r3_kernel",
    "fused_qkvzba_split_reshape_cat_gqa_r4_kernel",
};

}  // namespace

constexpr int32_t kDeviceId = 0;
// bfloat16 precision tolerance
constexpr float kTolerance = 2e-2f;

// ---------------------------------------------------------------------------
// CPU golden implementation
// Mirrors fused_qkvzba_split_reshape_cat_ref() from the Python test file.
// ---------------------------------------------------------------------------
struct SplitResult {
  torch::Tensor mixed_qkv;  // [B, nqk*hqk*2 + nv*hv]
  torch::Tensor z;          // [B, nv, hv]
  torch::Tensor b;          // [B, nv]
  torch::Tensor a;          // [B, nv]
};

static SplitResult fused_qkvzba_split_reshape_cat_ref(
    const torch::Tensor& mixed_qkvz,
    const torch::Tensor& mixed_ba,
    int64_t num_heads_qk,
    int64_t num_heads_v,
    int64_t head_qk,
    int64_t head_v) {
  const int64_t batch = mixed_qkvz.size(0);
  const int64_t v_heads_per_qk = num_heads_v / num_heads_qk;
  const int64_t qkvz_dim_t = head_qk * 2 + v_heads_per_qk * head_v * 2;
  const int64_t ba_dim_t = v_heads_per_qk * 2;

  // Work in float32 for reference computation to avoid precision loss
  auto qkvz = mixed_qkvz.reshape({batch, num_heads_qk, qkvz_dim_t}).to(torch::kFloat32);
  auto ba = mixed_ba.reshape({batch, num_heads_qk, ba_dim_t}).to(torch::kFloat32);

  // Split along the last dimension of qkvz
  auto q = qkvz.slice(/*dim=*/-1, 0, head_qk);
  auto k = qkvz.slice(-1, head_qk, 2 * head_qk);
  auto v = qkvz.slice(-1, 2 * head_qk, 2 * head_qk + v_heads_per_qk * head_v);
  auto z = qkvz.slice(-1, 2 * head_qk + v_heads_per_qk * head_v);

  auto b_ref = ba.slice(-1, 0, v_heads_per_qk);
  auto a_ref = ba.slice(-1, v_heads_per_qk);

  // Flatten and concatenate QKV
  auto mixed_qkv_ref = torch::cat(
      {
          q.reshape({batch, num_heads_qk * head_qk}),
          k.reshape({batch, num_heads_qk * head_qk}),
          v.reshape({batch, num_heads_v * head_v}),
      },
      /*dim=*/-1).to(mixed_qkvz.dtype());

  auto z_ref = z.reshape({batch, num_heads_v, head_v}).to(mixed_qkvz.dtype());
  auto b_out = b_ref.reshape({batch, num_heads_v}).to(mixed_ba.dtype());
  auto a_out = a_ref.reshape({batch, num_heads_v}).to(mixed_ba.dtype());

  return {mixed_qkv_ref, z_ref, b_out, a_out};
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------
// Test parameters: (batch, num_heads_qk, num_heads_v, head_qk, head_v)
using TestParams =
    std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>;

class TritonFusedQkvzbaSplitReshapeTest
    : public ::testing::TestWithParam<TestParams> {
 protected:
  static bool npu_initialized_;

  static void SetUpTestSuite() {
    try {
      torch::zeros({1}, torch::TensorOptions().device("npu:0"));
      c10_npu::SetDevice(kDeviceId);
      auto& reg = KernelRegistry::get_instance();
      npu_initialized_ = true;
      for (const char* kernel_name : kFusedQkvzbaSplitReshapeKernels) {
        std::string binary_path =
            GetKernelBinaryPath(std::string(kernel_name) + ".npubin");
        npu_initialized_ =
            npu_initialized_ &&
            reg.register_kernel(kernel_name, binary_path) &&
            reg.get_kernel_stub(kernel_name) != nullptr;
      }
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
    device_str_ = "npu:" + std::to_string(kDeviceId);
  }

  void TearDown() override {}

  bool npu_available_ = false;
  std::string device_str_;
};

bool TritonFusedQkvzbaSplitReshapeTest::npu_initialized_ = false;

// ---------------------------------------------------------------------------
// Test body
// ---------------------------------------------------------------------------
TEST_P(TritonFusedQkvzbaSplitReshapeTest, SplitReshapeKernelTest) {
  if (!npu_available_) {
    GTEST_SKIP() << "NPU device not available";
  }

  const int64_t batch = std::get<0>(GetParam());
  const int64_t num_heads_qk = std::get<1>(GetParam());
  const int64_t num_heads_v = std::get<2>(GetParam());
  const int64_t head_qk = std::get<3>(GetParam());
  const int64_t head_v = std::get<4>(GetParam());

  ASSERT_EQ(num_heads_v % num_heads_qk, 0)
      << "num_heads_v must be a multiple of num_heads_qk";

  const int64_t v_heads_per_qk = num_heads_v / num_heads_qk;
  const int64_t qkvz_dim =
      num_heads_qk * (head_qk * 2 + v_heads_per_qk * head_v * 2);
  const int64_t ba_dim = num_heads_qk * v_heads_per_qk * 2;

  auto device = at::Device(device_str_);
  auto opts_cpu_bf16 =
      torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU);

  // Create random input on CPU
  auto mixed_qkvz_cpu = torch::randn({batch, qkvz_dim}, opts_cpu_bf16);
  auto mixed_ba_cpu = torch::randn({batch, ba_dim}, opts_cpu_bf16);

  // Reference output
  auto ref = fused_qkvzba_split_reshape_cat_ref(
      mixed_qkvz_cpu, mixed_ba_cpu, num_heads_qk, num_heads_v, head_qk,
      head_v);

  // Run kernel on NPU
  auto mixed_qkvz_npu = mixed_qkvz_cpu.to(device).contiguous();
  auto mixed_ba_npu = mixed_ba_cpu.to(device).contiguous();

  auto [out_qkv, out_z, out_b, out_a] = npu_fused_qkvzba_split_reshape_cat(
      mixed_qkvz_npu, mixed_ba_npu,
      static_cast<int32_t>(num_heads_qk),
      static_cast<int32_t>(num_heads_v),
      static_cast<int32_t>(head_qk),
      static_cast<int32_t>(head_v));

  // Synchronise before comparing
  auto npu_stream = c10_npu::getCurrentNPUStream(kDeviceId);
  aclrtSynchronizeStream(npu_stream.stream());

  // Helper lambda for readable assertion
  auto check = [&](const torch::Tensor& golden,
                   const torch::Tensor& actual,
                   const char* name) {
    auto diff = torch::abs(golden - actual.cpu().contiguous());
    float max_diff = torch::max(diff).item<float>();
    EXPECT_LT(max_diff, kTolerance)
        << name << " mismatch: max_diff=" << max_diff
        << " > tolerance=" << kTolerance
        << " (batch=" << batch << " nqk=" << num_heads_qk
        << " nv=" << num_heads_v << " hqk=" << head_qk << " hv=" << head_v
        << ")";
  };

  check(ref.mixed_qkv, out_qkv, "mixed_qkv");
  check(ref.z, out_z, "z");
  check(ref.b, out_b, "b");
  check(ref.a, out_a, "a");
}

// ---------------------------------------------------------------------------
// Instantiate test cases
// Covers typical Qwen3-Next GatedDeltaNet configurations
// ---------------------------------------------------------------------------
INSTANTIATE_TEST_SUITE_P(
    FusedQkvzbaSplitReshapeParams,
    TritonFusedQkvzbaSplitReshapeTest,
    ::testing::Values(
        // (batch, num_heads_qk, num_heads_v, head_qk, head_v)
        std::make_tuple(1,  8, 16, 128, 128),
        std::make_tuple(4,  8, 16, 128, 128),
        std::make_tuple(8,  8, 16, 128, 128),
        std::make_tuple(1, 16, 16, 128, 128),
        std::make_tuple(1, 16, 32, 128, 128),
        std::make_tuple(1, 16, 48, 128, 128),
        std::make_tuple(1, 16, 64, 128, 128),
        std::make_tuple(4, 16, 32, 128, 128),
        std::make_tuple(8, 16, 32, 128, 128),
        // Qwen3.5 0.8B / 2B local heads: 16->16 split by TP1/2/4/8.
        std::make_tuple(64, 16, 16, 128, 128),
        std::make_tuple(64,  8,  8, 128, 128),
        std::make_tuple(64,  4,  4, 128, 128),
        std::make_tuple(64,  2,  2, 128, 128),
        // Qwen3.5 9B / Qwen3.6 35B-A3B local heads: 16->32 split by TP1/2/4/8.
        std::make_tuple(64, 16, 32, 128, 128),
        std::make_tuple(64,  8, 16, 128, 128),
        std::make_tuple(64,  4,  8, 128, 128),
        std::make_tuple(64,  2,  4, 128, 128),
        // Qwen3.5 27B / Qwen3.6 27B local heads: 16->48 split by TP1/2/4/8.
        std::make_tuple(64, 16, 48, 128, 128),
        std::make_tuple(64,  8, 24, 128, 128),
        std::make_tuple(64,  4, 12, 128, 128),
        std::make_tuple(64,  2,  6, 128, 128),
        // Qwen3.5 397B-A17B / 122B-A10B local heads: 16->64 split by TP1/2/4/8.
        std::make_tuple(64, 16, 64, 128, 128),
        std::make_tuple(64,  8, 32, 128, 128),
        std::make_tuple(64,  4, 16, 128, 128),
        std::make_tuple(64,  2,  8, 128, 128),
        // Larger batches to exercise >65535-token code path
        std::make_tuple(4096,  2,  4, 128, 128),
        std::make_tuple(4096, 16, 16, 128, 128),
        std::make_tuple(4096,  4,  8, 128, 128),
        std::make_tuple(4096,  8,  16, 128, 128),
        std::make_tuple(4096, 16,  32, 128, 128),
        std::make_tuple(4096, 16,  48, 128, 128),
        std::make_tuple(4096, 16,  64, 128, 128),
        std::make_tuple(256,  8, 16, 128, 128)));

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
