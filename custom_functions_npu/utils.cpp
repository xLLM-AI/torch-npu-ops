#include "utils.h"

#include <torch_npu/csrc/core/npu/DeviceUtils.h>

namespace atb {
namespace utils {

ContextManager& ContextManager::get_instance() {
  static ContextManager instance;
  return instance;
}

ContextManager::ContextManager() {}

ContextManager::~ContextManager() {
  for (auto& entry : device_contexts_) {
    if (entry.second != nullptr) {
      auto status = atb::DestroyContext(entry.second);
      CHECK_EQ(status, 0) << "Destroy context failed!";
    }
  }
  device_contexts_.clear();
}

atb::Context* ContextManager::get_context(aclrtStream stream) {
  // ATB contexts are bound to the device current at creation time, so keep one
  // per device. The calling thread runs on its worker's device (set via
  // aclrtSetDevice), so query that device and create/reuse its context.
  int32_t device_id = 0;
  auto device_status = aclrtGetDevice(&device_id);
  CHECK_EQ(device_status, ACL_ERROR_NONE) << "aclrtGetDevice failed!";

  atb::Context* context = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_contexts_.find(device_id);
    if (it == device_contexts_.end()) {
      auto status = atb::CreateContext(&context);
      CHECK_EQ(status, 0) << "Create context failed!";
      device_contexts_[device_id] = context;
    } else {
      context = it->second;
    }
  }

  context->SetExecuteStream(stream);
  return context;
}

atb::Context* get_context(aclrtStream stream) {
  return ContextManager::get_instance().get_context(stream);
}

aclDataType convert_to_acl_data_type(const at::ScalarType& data_type) {
  auto acl_dtype =
      kATenScalarTypeToAclDataTypeTable[static_cast<int64_t>(data_type)];
  CHECK_NE(acl_dtype, ACL_DT_UNDEFINED)
      << std::string(c10::toString(data_type)) << " has not been supported";
  return acl_dtype;
}

at::Tensor format_trans(const at::Tensor& at_tensor) {
  if (torch_npu::utils::is_npu(at_tensor)) {
    return at_npu::native::npu_format_cast(at_tensor, ACL_FORMAT_ND);
  }
  return at_tensor;
}

bool is_base_format(aclFormat& format) {
  return (format == ACL_FORMAT_NCHW) || (format == ACL_FORMAT_ND) ||
         (format == ACL_FORMAT_NHWC) || (format == ACL_FORMAT_NCDHW);
}

aclFormat get_format_for_atb(const at::Tensor& at_tensor) {
  if (torch_npu::utils::is_npu(at_tensor)) {
    aclFormat format =
        static_cast<aclFormat>(at_npu::native::get_npu_format(at_tensor));
    return is_base_format(format) ? ACL_FORMAT_ND : format;
  }
  return ACL_FORMAT_ND;
}
}  // namespace utils
}  // namespace atb
