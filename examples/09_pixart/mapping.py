from typing import Dict, Optional, Union

import torch

from aitemplate.compiler.dtype import _DTYPE_TO_TORCH_DTYPE

from config import Transformer2DModelConfig


def transformer2d_mapping(
    state_dict: Dict[str, torch.Tensor],
    config: Transformer2DModelConfig,
    device: Optional[Union[str, torch.device]] = "cuda",
    dtype: Optional[str] = "float16",
):
    assert (
        dtype in _DTYPE_TO_TORCH_DTYPE
    ), f"Unknown dtype: {dtype}. Expected one of {_DTYPE_TO_TORCH_DTYPE.keys()}"
    params = {}
    torch_dtype = _DTYPE_TO_TORCH_DTYPE[dtype]
    for key in list(state_dict.keys()):
        ait_key = key.replace(".", "_")
        shape = state_dict[key].shape
        if len(shape) == 4 and "weight" in key and "norm" not in key:
            params[ait_key] = state_dict[key].permute(0, 2, 3, 1).contiguous()
        else:
            params[ait_key] = state_dict[key]
    params = {k: v.to(device, torch_dtype) for k, v in params.items()}
    # TODO: where does this come from?
    dim = 256
    params["time_proj"] = torch.arange(start=0, end=dim // 2, dtype=torch.float32).to(
        device, dtype=torch_dtype
    )
    if config.use_additional_conditions:
        params["additional_condition_proj"] = torch.arange(
            start=0, end=dim // 2, dtype=torch.float32
        ).to(device, dtype=torch_dtype)
    return params
