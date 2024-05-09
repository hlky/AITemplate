from typing import cast

import click
import torch

from aitemplate.compiler import Model
from aitemplate.testing.benchmark_ait import benchmark_module
from aitemplate.testing.benchmark_pt import benchmark_torch_function
from config import get_config

from diffusers import Transformer2DModel

from mapping import transformer2d_mapping


@click.command()
@click.option(
    "--hf-hub",
    default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    help="hf hub model name",
)
@click.option("--work-dir", default="./tmp", help="Work directory")
@click.option("--model-name", default="Transformer2DModel", help="Model name")
def benchmark_transformer2d(
    hf_hub,
    work_dir="./tmp",
    model_name="Transformer2DModel",
):
    config = get_config(hf_hub=hf_hub)

    pt_model = cast(
        Transformer2DModel,
        Transformer2DModel.from_pretrained(
            hf_hub,
            torch_dtype=torch.float16,
            use_safetensors=None,
            subfolder="transformer",
        ),
    )
    pt_model = pt_model.eval().cuda().half()

    latent_input = torch.randn(1, 4, 128, 128).cuda().half()
    encoder_hidden_states = torch.randn(1, 128, config.caption_channels).cuda().half()
    timestep = torch.tensor([999]).cuda().long()
    added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

    pt_t = benchmark_torch_function(
        25,
        pt_model.forward,
        latent_input,
        encoder_hidden_states,
        timestep,
        added_cond_kwargs,
    )
    print(f"PyTorch: {pt_t:.3f} ms")

    state_dict = dict(pt_model.state_dict())
    device = "cuda"
    dtype = "float16"

    params = transformer2d_mapping(state_dict, config, device, dtype)

    module = Model(f"{work_dir}/{model_name}/test.so")
    failed = False
    for key, tensor in params.items():
        try:
            module.set_constant_with_tensor(key, tensor)
        except Exception as e:
            print(f"Failed to set constant {key}: {e}")
            failed = True

    if failed:
        return

    module.fold_constants(sync=True)

    benchmark_module(module)


if __name__ == "__main__":
    benchmark_transformer2d()
