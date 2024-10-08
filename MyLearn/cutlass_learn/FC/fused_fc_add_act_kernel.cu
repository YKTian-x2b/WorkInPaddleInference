// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <glog/logging.h>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/fusion/cutlass/fully_connected/fc_decl.h"

#include "paddle/phi/backends/dynload/cutlass_fc.h"

namespace phi {
namespace fusion {
namespace cutlass_internal {

typedef void (*func)(phi::fusion::cutlass_internal::FcAllParams);

template <typename T, typename Context>
void FusedFcAddActKernel(const Context& ctx,
                             const DenseTensor& input,
                             const DenseTensor& weight,
                             const DenseTensor& bias,
                             const std::string& data_format,
                             const std::string& activation,
                             float alpha,
                             DenseTensor* output) {
  ctx.template Alloc<T>(output);
  auto input_dims = input.dims();
  auto weight_dims = weight.dims();
  auto bias_dims = bias.dims();
  auto out_dims = output->dims();

  CHECK_EQ(input_dims.size() == 2UL, true);
  CHECK_EQ(weight_dims.size() == 2UL, true);
  CHECK_EQ(bias_dims.size() == 1UL, true);
  CHECK_EQ(out_dims.size() == 2UL, true);

  CHECK_EQ(data_format == "RRR", true);
  const int M = input_dims[0];
  const int K = input_dims[1];
  const int N = weight_dims[1];
  // 在RRR的情况下
  const int lda = K;
  const int ldb = N;
  const int ldd = N;

  int64_t device_id = ctx.GetPlace().GetDeviceId();
  int sm_version = backends::gpu::GetGPUComputeCapability(device_id);

  // 这里是否考虑 output和weight的数据类型？
  auto get_fc_dtype = [&](decltype(input.dtype()) input_type)
      -> phi::fusion::cutlass_internal::FcDataType {
    switch (input_type) {
      case phi::DataType::FLOAT32:
        return FcDataType::fp32;
      case phi::DataType::FLOAT16:
        return FcDataType::fp16;
      case phi::DataType::BFLOAT16:
        return FcDataType::bf16;
    }
  };

  auto cutlass_dispatch_sm_version = [&](int device_sm_version) -> int {
    if (device_sm_version < 75) {
      PADDLE_ENFORCE_GE(
          device_sm_version,
          75,
          phi::errors::PreconditionNotMet(
              "fused_fc_add_act only supports sm >= 75, but got %d.",
              device_sm_version));
    } else if (device_sm_version > 80) {
      return 80;
    } else {
      return device_sm_version;
    }
  };

  FcAllParams params = {
      reinterpret_cast<const void*>(input.data<T>()),
      reinterpret_cast<const void*>(weight.data<T>()),
      reinterpret_cast<const void*>(bias.data<T>()),
      reinterpret_cast<void*>(output->data<T>()),
      M,
      N,
      K,
      lda, 
      ldb,
      ldd,
      alpha,    // alpha
      ctx.stream(),
      get_fc_dtype(input.dtype()),
      cutlass_dispatch_sm_version(sm_version),
      0.,       // beta
  };

  void* dlhandler = phi::dynload::GetCutlassFcHandle();
  func fc_func = NULL;
  CHECK_EQ(dlhandler == NULL, false);

  if (activation == "relu") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasRelu"));
  } else if (activation == "swish") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasSilu"));
  } else if (activation == "identity") {
    fc_func = (func)(dlsym(dlhandler, "FcBias"));
  } else if (activation == "leaky_relu") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasLeakyRelu"));
    params.alpha = fuse_alpha;
  } else if (activation == "sigmoid") {
    fc_func = (func)(dlsym(dlhandler, "FcBiasSigmoid"));
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Cutlass does not support this activation: %s.", activation.c_str()));
  }
  fc_func(params);
}
}  // namespace cutlass_internal
}  // namespace fusion
}  // namespace phi

PD_REGISTER_KERNEL(fused_fc_add_act,
                   GPU,
                   ALL_LAYOUT,
                   phi::fusion::cutlass_internal::FusedFcAddActKernel,
                   float,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
