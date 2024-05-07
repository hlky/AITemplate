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
import click
import numpy as np
import safetensors.torch
import torch

from aitemplate.compiler import Model
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from PIL import Image
from modeling.rrdbnet_pt import RRDBNet


def esrgan_inference(
    exe_module: Model,
    input_pixels: np.ndarray,
    scale=4,
    benchmark: bool = False,
) -> torch.Tensor | float:
    if np.max(input_pixels) > 256:
        max_range = 65535
    else:
        max_range = 255
    input_pixels = input_pixels / max_range
    height, width, _ = input_pixels.shape
    inputs = {
        "input_pixels": torch.from_numpy(input_pixels)
        .unsqueeze(0)
        .contiguous()
        .cuda()
        .half(),
    }
    ys = {}
    for name, idx in exe_module.get_output_name_to_index_map().items():
        shape = exe_module.get_output_maximum_shape(idx)
        shape[1] = height * scale
        shape[2] = width * scale
        ys[name] = torch.empty(shape).cuda().half()
    if benchmark:
        t, _, _ = exe_module.benchmark_with_tensors(inputs, ys, count=25, repeat=2)
        return t
    else:
        exe_module.run_with_tensors(inputs, ys, graph_mode=False)
        upscaled = ys["upscaled_pixels"]
        upscaled = upscaled.squeeze(0).cpu().clamp_(0, 1).numpy()
        if max_range == 65535:
            upscaled = (upscaled * 65535.0).round().astype(np.uint16)
        else:
            upscaled = (upscaled * 255.0).round().astype(np.uint8)
        return upscaled


@click.command()
@click.option(
    "--module-path",
    default="./tmp/ESRGANModel/test.so",
    help="the AIT module path",
)
@click.option(
    "--input-image-path",
    default="input.png",
    help="path to input image",
)
@click.option(
    "--output-image-path",
    default="output.png",
    help="path to output image",
)
@click.option(
    "--scale",
    default=4,
    help="Scale of ESRGAN model",
)
@click.option(
    "--benchmark",
    is_flag=True,
    help="Benchmark mode",
)
@click.option(
    "--model-path",
    default="RealESRGAN_x4plus.pth",
    help="model path. supports torch or safetensors",
)
@click.option(
    "--size",
    default=256,
    help="size of input image",
)
def demo(
    module_path,
    input_image_path,
    output_image_path,
    scale,
    benchmark,
    model_path,
    size,
):
    module = Model(module_path)
    input_image = Image.open(input_image_path).convert("RGB").resize((size, size))
    image_array = np.array(input_image)

    upscaled = esrgan_inference(module, image_array, scale, benchmark=benchmark)
    if benchmark:
        ait_t = upscaled
    else:
        output_image = Image.fromarray(upscaled)
        output_image.save(output_image_path)
        return

    if model_path.endswith(".safetensors"):
        pt_model = safetensors.torch.load_file(model_path)
    else:
        pt_model = torch.load(model_path)

    if "params_ema" in pt_model.keys():
        pt_model = pt_model["params_ema"]
    elif "params" in pt_model.keys():
        pt_model = pt_model["params"]

    del module
    pt_module = RRDBNet(3, 3, scale=scale)
    pt_module.load_state_dict(pt_model)
    pt_module.eval()
    pt_module = pt_module.cuda().half()
    pt_input = torch.from_numpy(image_array).unsqueeze(0).permute((0, 3, 1, 2)).contiguous().cuda().half()
    pt_t = benchmark_torch_function(25, pt_module.forward, pt_input)
    print(f"AIT: {ait_t:.2f} ms")
    print(f"PyTorch: {pt_t:.2f} ms")



if __name__ == "__main__":
    demo()
