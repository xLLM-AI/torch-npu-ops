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

#pragma once

#include <acl/acl.h>
#include <dlfcn.h>
#include <glog/logging.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <system_error>
#include <vector>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "args_builder.h"
#include "kernel_registry.h"
#include <iostream> 
namespace xllm::kernel::npu {

inline bool is_regular_file_path(const std::filesystem::path& path) {
  std::error_code error_code;
  return std::filesystem::is_regular_file(path, error_code);
}

inline bool is_ascend950_soc() {
  static const bool is_ascend950 = []() {
    const char* soc_name = aclrtGetSocName();
    return soc_name != nullptr &&
           std::string(soc_name).find("Ascend950") != std::string::npos;
  }();
  return is_ascend950;
}

inline void append_unique_path(std::vector<std::filesystem::path>* paths,
                               const std::filesystem::path& path) {
  if (path.empty()) {
    return;
  }

  std::filesystem::path normalized_path = path.lexically_normal();
  for (const auto& existing_path : *paths) {
    if (existing_path == normalized_path) {
      return;
    }
  }
  paths->push_back(std::move(normalized_path));
}

inline void append_env_binary_root(std::vector<std::filesystem::path>* paths,
                                   const char* env_name) {
  const char* env_value = std::getenv(env_name);
  if (env_value == nullptr || env_value[0] == '\0') {
    return;
  }
  append_unique_path(paths, std::filesystem::path(env_value));
}

inline std::filesystem::path get_current_object_dir() {
  Dl_info dl_info{};
  if (dladdr(reinterpret_cast<const void*>(&get_current_object_dir),
             &dl_info) == 0 ||
      dl_info.dli_fname == nullptr) {
    return {};
  }

  return std::filesystem::path(dl_info.dli_fname).parent_path();
}

inline std::filesystem::path get_executable_dir() {
  std::vector<char> buffer(/*capacity=*/4096, '\0');
  ssize_t length = readlink("/proc/self/exe", buffer.data(), buffer.size() - 1);
  if (length <= 0) {
    return {};
  }

  return std::filesystem::path(std::string(buffer.data(), length))
      .parent_path();
}

inline std::vector<std::filesystem::path> get_candidate_binary_roots() {
  std::vector<std::filesystem::path> roots;

  append_env_binary_root(&roots, "TRITON_BINARY_PATH");

#ifdef TRITON_BINARY_PATH
  append_unique_path(&roots, std::filesystem::path(TRITON_BINARY_PATH));
#endif

  std::filesystem::path current_object_dir = get_current_object_dir();
  if (!current_object_dir.empty()) {
    append_unique_path(&roots, current_object_dir / "triton_npu" / "binary");
  }

  std::filesystem::path executable_dir = get_executable_dir();
  if (!executable_dir.empty()) {
    append_unique_path(&roots, executable_dir / "triton_npu" / "binary");
  }

  return roots;
}

inline std::string resolve_npubin_path_by_kernel(const std::string& kernel_name) {

  std::string kernel_file_name = kernel_name + ".npubin";
  for (const auto& binary_root : get_candidate_binary_roots()) {
    std::filesystem::path candidate_path = binary_root / kernel_file_name;
    if (is_regular_file_path(candidate_path)) {
      return candidate_path.string();
    }
  }
#ifdef TRITON_BINARY_PATH
    return (std::filesystem::path(TRITON_BINARY_PATH) / kernel_file_name)
        .string();
#else
    return {};
#endif
}

class OperationBase {
 public:
  explicit OperationBase(std::string kernel_name, std::string npubin_path = "")
      : kernel_name_(std::move(kernel_name)),
        npubin_path_(std::move(npubin_path)) {}

  virtual ~OperationBase() = default;

  template <class BuildArgsFn>
  rtError_t execute(rtStream_t stream,
                    int32_t gridX,
                    int32_t gridY,
                    int32_t gridZ,
                    BuildArgsFn&& build_args) {
    aclmdlRICaptureStatus capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    aclmdlRI model_ri = nullptr;
    (void)aclmdlRICaptureGetInfo(stream, &capture_status, &model_ri);
    if (!ensure_registered()) {
      return static_cast<rtError_t>(-1);
    }

    const uint32_t block_num = static_cast<uint32_t>(gridX) *
                               static_cast<uint32_t>(gridY) *
                               static_cast<uint32_t>(gridZ);

    void* ffts_addr = nullptr;
    uint32_t ffts_len = 0;
    // Ascend950 Triton binaries omit the legacy hardware-sync argument.
    bool has_hardware_sync_arg = !is_ascend950_soc();
    rtError_t rt_ret = RT_ERROR_NONE;
    if (has_hardware_sync_arg) {
      rt_ret =
          rtGetC2cCtrlAddr(reinterpret_cast<uint64_t*>(&ffts_addr), &ffts_len);
      if (rt_ret != RT_ERROR_NONE) {
        const auto acl_sync_ret = aclrtGetHardwareSyncAddr(&ffts_addr);
        if (acl_sync_ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
          static std::once_flag warning_once;
          std::call_once(warning_once, [rt_ret]() {
            LOG(WARNING) << "Hardware sync address is unavailable; continuing "
                         << "without it (rtGetC2cCtrlAddr returned " << rt_ret
                         << ")";
          });
          ffts_addr = nullptr;
          has_hardware_sync_arg = false;
        } else if (acl_sync_ret != ACL_ERROR_NONE) {
          LOG(ERROR) << "Failed to get hardware sync address: rt=" << rt_ret
                     << ", acl=" << acl_sync_ret;
          return static_cast<rtError_t>(acl_sync_ret);
        }
      }
    }

    void* workspace = nullptr;
    void* lock = nullptr;
    const auto acl_ret = setup_workspace(block_num, &workspace, &lock);
    if (acl_ret != ACL_ERROR_NONE) {
      return static_cast<rtError_t>(acl_ret);
    }

    ArgsBuilder ab;
    if (has_hardware_sync_arg) {
      ab.add_aligned<void*>(ffts_addr, 8);
    }
    ab.add_aligned<void*>(lock, 8);
    ab.add_aligned<void*>(workspace, 8);
    build_args(ab);
    ab.add_aligned<int32_t>(gridX, 4);
    ab.add_aligned<int32_t>(gridY, 4);
    ab.add_aligned<int32_t>(gridZ, 4);

    KernelStubHandle stub =
        KernelRegistry::get_instance().get_kernel_stub(kernel_name_);
    if (stub == nullptr) {
      LOG(ERROR) << "Kernel stub is null for '" << kernel_name_ << "'";
      return static_cast<rtError_t>(-1);
    }

    rt_ret = rtKernelLaunch(stub,
                            block_num,
                            const_cast<void*>(ab.data()),
                            static_cast<uint32_t>(ab.size()),
                            nullptr,
                            stream);
    return rt_ret;
  }

 protected:
  const std::string& kernel_name() const { return kernel_name_; }

  virtual std::string resolve_npubin_path() const {
    if (!npubin_path_.empty()) {
      return npubin_path_;
    }

    std::string kernel_file_name = kernel_name_ + ".npubin";
    for (const auto& binary_root : get_candidate_binary_roots()) {
      std::filesystem::path candidate_path = binary_root / kernel_file_name;
      if (is_regular_file_path(candidate_path)) {
        return candidate_path.string();
      }
    }

#ifdef TRITON_BINARY_PATH
    return (std::filesystem::path(TRITON_BINARY_PATH) / kernel_file_name)
        .string();
#else
    return {};
#endif
  }

  bool ensure_registered() {
    auto& reg = KernelRegistry::get_instance();
    if (reg.is_kernel_registered(kernel_name_)) {
      return true;
    }

    const std::string bin = resolve_npubin_path();
    if (bin.empty()) {
      LOG(ERROR) << "Empty npubin path for kernel '" << kernel_name_ << "'";
      return false;
    }
    if (!reg.register_kernel(kernel_name_, bin)) {
      LOG(ERROR) << "Failed to register kernel '" << kernel_name_ << "' from "
                 << bin;
      return false;
    }
    return true;
  }

  aclError setup_workspace(uint32_t block_num, void** workspace, void** lock) {
    *workspace = nullptr;
    *lock = nullptr;

    int64_t workspace_size = -1;
    int64_t lock_init_value = 0;
    int64_t lock_num = -1;

    auto& reg = KernelRegistry::get_instance();
    reg.get_kernel_workspace_config(
        kernel_name_, workspace_size, lock_init_value, lock_num);

    at::TensorOptions options =
        at::TensorOptions(torch_npu::utils::get_npu_device_type());
    if (workspace_size > 0) {
      workspace_size *= static_cast<int64_t>(block_num);
      *workspace = const_cast<void *>(
        at::empty({workspace_size}, options.dtype(at::kByte)).storage().data());
    }

    if (lock_num > 0) {
      const uint64_t bytes = static_cast<uint64_t>(lock_num) * sizeof(int64_t);
      *lock = const_cast<void *>(
        at::empty({bytes}, options.dtype(at::kByte)).storage().data());

      std::vector<int64_t> init(static_cast<size_t>(lock_num), lock_init_value);
      auto ret = aclrtMemcpy(
          *lock, bytes, init.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
      if (ret != ACL_ERROR_NONE) {
        LOG(ERROR) << "aclrtMemcpy lock init failed for '" << kernel_name_
                   << "': " << ret;
        return ret;
      }
    }

    return ACL_ERROR_NONE;
  }

  void cleanup_workspace(void* workspace, void* lock) {
    if (workspace) {
      aclrtFree(workspace);
    }
    if (lock) {
      aclrtFree(lock);
    }
  }

  struct PendingRelease {
    void* workspace = nullptr;
    void* lock = nullptr;
    aclrtEvent event = nullptr;
  };

  void cleanup_completed_releases() {
    std::lock_guard<std::mutex> guard(pending_mu_);
    auto it = pending_releases_.begin();
    while (it != pending_releases_.end()) {
      if (it->event == nullptr) {
        // Already freed or fallback path
        it = pending_releases_.erase(it);
        continue;
      }

      aclrtEventStatus status = ACL_EVENT_STATUS_NOT_READY;
      auto ret = aclrtQueryEvent(it->event, &status);
      if (ret != ACL_ERROR_NONE) {
        LOG(WARNING) << "aclrtQueryEvent failed for '" << kernel_name_
                     << "': " << ret;
        ++it;
        continue;
      }

      if (status == ACL_EVENT_STATUS_COMPLETE) {
        if (it->workspace) {
          aclrtFree(it->workspace);
        }
        if (it->lock) {
          aclrtFree(it->lock);
        }
        aclrtDestroyEvent(it->event);
        it = pending_releases_.erase(it);
      } else {
        ++it;
      }
    }
  }

 private:
  std::string kernel_name_;
  std::string npubin_path_;
  std::mutex pending_mu_;
  std::vector<PendingRelease> pending_releases_;
};

}  // namespace xllm::kernel::npu
