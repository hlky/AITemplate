#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
"""
Common codegen functions for gemm with bias.
"""

import jinja2

# pylint: disable=C0301,C0415,R1705

INSTANCE_TEMPLATE = jinja2.Template(
    """
{{config}}
using {{name}} = {{config_name}};
"""
)


SRC_TEMPLATE = jinja2.Template(
    """
{% if func_only %}
{{instances}}
{% else %}
#include <iostream>
#include <memory>
#include <random>
#include <vector>
#include "short_file.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/epilogue/thread/linear_combination_silu.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/device_memory.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass/epilogue/thread/linear_combination_residual_block.h"
#include "cutlass/gemm/device/gemm_universal_with_broadcast.h"


namespace cutlass {
namespace epilogue {
namespace thread {

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationFastGELU = LinearCombinationGeneric<GELU_taylor, ElementOutput_, Count, ElementAccumulator_,
                                                          ElementCompute_, Scale, Round, true>;

template <
  typename ElementOutput_,                             ///< Data type used to load and store tensors
  int Count,                                           ///< Number of elements computed per operation
                                                       ///< Usually it is 128/sizeof_bits<ElementOutput_>,
                                                       ///< but we use 64 or 32 sometimes when there are not enough data to store
  typename ElementAccumulator_ = ElementOutput_,       ///< Accumulator data type
  typename ElementCompute_ = ElementOutput_,           ///< Data type used to compute linear combination
  ScaleType::Kind Scale = ScaleType::Default,          ///< Control Alpha and Beta scaling
  FloatRoundStyle Round = FloatRoundStyle::round_to_nearest
>
using LinearCombinationTanh = LinearCombinationGeneric<Tanh, ElementOutput_, Count, ElementAccumulator_,
                                                          ElementCompute_, Scale, Round, true>;

} // namespace thread
} // namespace epilogue
} // namespace cutlass

using bfloat16 = nv_bfloat16;

{{extra_code}}

#define CUTLASS_CHECK(status)                                                         \\
  {                                                                                   \\
    cutlass::Status error = status;                                                   \\
    if (error != cutlass::Status::kSuccess) {                                         \\
      auto msg = std::string("[") + __SHORT_FILE__ + "] Got cutlass error: " +              \\
          cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \\
      std::cerr << msg << std::endl;                                                  \\
      throw std::runtime_error(msg);                                                  \\
    }                                                                                 \\
  }

{{instances}}
{% endif %}

{% if is_profiler %}
template <typename GemmInstance>
void {{function_name}} (
    GemmInstance& gemm_op,
{% else %}
void {{function_name}} (
{% endif %}
    void* a_ptr,
    void* b_ptr,
    void* bias_ptr,
    void* c_ptr,
    uint8_t* workspace,
{% if support_split_k %}
    int split_k,
{% endif %}
{% for idx in range(input_ndims) %}
    int64_t* a_dim{{idx}},
{% endfor %}
{% for idx in range(weight_ndims) %}
    int64_t* b_dim{{idx}},
{% endfor %}
{% for idx in range(input_ndims) %}
    int64_t* c_dim{{idx}},
{% endfor %}
  cudaStream_t stream
  ) {
  {{shape_eval}}
  {{input_addr_calculator}}
  {{output_addr_calculator}}
  {{extra_shape}}
  {{input_output_checks}}

  if (!bias_ptr) {
    throw std::runtime_error("bias_ptr is null!");
  }

  {{exec_paths}}
  {% for idx in range(input_ndims) %}
      std::cout << "input_ndims{{idx}}: " << *a_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(weight_ndims) %}
      std::cout << "weight_ndims{{idx}}: " << *b_dim{{idx}} << std::endl;
  {% endfor %}
  {% for idx in range(input_ndims) %}
      std::cout << "output_ndims{{idx}}: " << *c_dim{{idx}} << std::endl;
  {% endfor %}
  throw std::runtime_error(
      "Unsupported workload for this {{function_name}} specialization."
  );
}
""",
    trim_blocks=True,
    lstrip_blocks=True,
)


FUNC_DECL_TEMPLATE = jinja2.Template(
    """
void {{func_name}}(
  void*,
  void*,
  void*,
  void*,
  uint8_t*,
{% if support_split_k %}
    int,
{% endif %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(weight_ndims) %}
  int64_t*,
{% endfor %}
{% for idx in range(input_ndims) %}
  int64_t*,
{% endfor %}
  cudaStream_t
);
"""
)
