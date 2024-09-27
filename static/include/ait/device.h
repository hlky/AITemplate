#pragma once
#ifdef AIT_CUDA
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"
#endif
#ifdef AIT_HIP
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "library/include/ck/library/utility/host_tensor.hpp"
#endif

namespace ait {
#ifdef AIT_CUDA
using bfloat16 = __nv_bfloat16;
using DeviceStream = cudaStream_t;
#endif
#ifdef AIT_HIP
using DeviceStream = hipStream_t;
#endif
} // namespace ait