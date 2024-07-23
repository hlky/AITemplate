from aitemplate.backend import registry
from aitemplate.backend.backend_spec import CUDASpec
from aitemplate.backend.common import pad_common

Header_Files = """
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "cutlass/util/host_tensor.h"

using bfloat16 = __nv_bfloat16;
"""


@registry.reg("cuda.pad.gen_function")
def gen_function(func_attrs, template_path):
    func_name = func_attrs["name"]
    backend_spec = CUDASpec()
    input_type = backend_spec.dtype_to_backend_type(
        func_attrs["inputs"][0]._attrs["dtype"]
    )
    return pad_common.SRC_TEMPLATE.render(
        header_files=Header_Files,
        function_name=func_name,
        index_type=backend_spec.index_type,
        prefix=backend_spec.prefix,
        elem_input_type=input_type,
        elem_output_type=input_type,
    )


@registry.reg("cuda.pad.func_decl")
def pad_gen_function_decl(func_attrs):
    return pad_common.gen_function_decl(func_attrs, backend_spec=CUDASpec())


@registry.reg("cuda.pad.func_call")
def pad_gen_function_call(func_attrs, indent="    "):
    return pad_common.gen_function_call(
        func_attrs, backend_spec=CUDASpec(), indent=indent
    )
