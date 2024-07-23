import unittest

import torch

from aitemplate.compiler import compile_model, ops
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target
from aitemplate.testing.test_utils import (
    filter_test_cases_by_params,
    get_random_torch_tensor,
    TestEnv,
)
from parameterized import parameterized


_DEFAULT_BATCH_SIZE = [1, 3]


class PixelUnshuffleTestCase(unittest.TestCase):
    def _test_single_op(
        self,
        downscale_factor=2,
        batch_size=_DEFAULT_BATCH_SIZE,
        test_name="pixel_unshuffle_fp16",
        dtype="float16",
    ):
        channels = 1024
        HH, WW = 8 * downscale_factor, 8 * downscale_factor
        target = detect_target()
        X = Tensor(
            shape=[IntVar(values=batch_size, name="input_batch"), HH, WW, channels],
            dtype=dtype,
            name="input_0",
            is_input=True,
        )
        OP = ops.pixel_unshuffle(downscale_factor)
        Y = OP(X)
        Y._attrs["name"] = "output_0"
        Y._attrs["is_output"] = True
        module = compile_model(Y, target, "./tmp", test_name)

        for b in batch_size:
            X_pt = get_random_torch_tensor([b, channels, HH, WW], dtype=dtype)
            Y_pt = torch.nn.functional.pixel_unshuffle(X_pt, downscale_factor)
            x = torch.permute(X_pt, (0, 2, 3, 1)).contiguous()
            y = torch.empty_like(Y_pt).permute((0, 2, 3, 1)).contiguous()
            module.run_with_tensors([x], [y])
            y_transpose = torch.permute(y, (0, 3, 1, 2))
            torch.testing.assert_close(
                y_transpose,
                Y_pt.to(y.dtype),
                rtol=1e-3,
                atol=1e-3,
                msg=lambda msg: f"{msg}\n\n{test_name}\npt ({Y_pt.shape}):\n{Y_pt}\n\nait ({y_transpose.shape}):\n{y_transpose}\n\n",
            )

    @parameterized.expand(
        **filter_test_cases_by_params(
            {
                TestEnv.CUDA_LESS_THAN_SM80: [("float16"), ("float32")],
                TestEnv.CUDA_SM80: [("bfloat16")],
                TestEnv.ROCM: [("float16")],
            }
        )
    )
    def test_pixel_unshuffle(self, ait_dtype):
        self._test_single_op(
            downscale_factor=2,
            test_name=f"pixel_unshuffle_{ait_dtype}",
            dtype=ait_dtype,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    unittest.main()