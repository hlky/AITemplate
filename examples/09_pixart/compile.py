import logging

import click
import torch

from aitemplate.compiler import compile_model
from aitemplate.frontend import IntVar, Tensor
from aitemplate.testing import detect_target

from config import get_config
from modeling.transformer_2d import Transformer2DModel


def mark_output(tensor: Tensor, name: str):
    tensor._attrs["is_output"] = True
    tensor._attrs["name"] = name
    shape = [d._attrs["values"] for d in tensor._attrs["shape"]]
    print(f"AIT output `{name}` shape {shape}")
    return tensor


@click.command()
@click.option(
    "--hf-hub",
    default="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
    help="hf hub model name",
)
@click.option(
    "--width",
    default=(1024, 1024),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum width",
)
@click.option(
    "--height",
    default=(1024, 1024),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum height",
)
@click.option(
    "--batch-size",
    default=(1, 1),
    type=(int, int),
    nargs=2,
    help="Minimum and maximum batch size",
)
@click.option(
    "--include-constants",
    default=True,
    type=bool,
    help="include constants (model weights) with compiled model",
)
@click.option("--use-fp16-acc", default=True, help="use fp16 accumulation")
@click.option("--convert-conv-to-gemm", default=True, help="convert 1x1 conv to gemm")
@click.option("--work-dir", default="./tmp", help="Work directory")
@click.option("--model-name", default="Transformer2DModel", help="Model name")
def compile_transformer2d(
    hf_hub,
    width,
    height,
    batch_size,
    include_constants,
    use_fp16_acc=True,
    convert_conv_to_gemm=True,
    work_dir="./tmp",
    model_name="Transformer2DModel",
):
    logging.getLogger().setLevel(logging.INFO)
    torch.manual_seed(69420)

    if detect_target().name() == "rocm":
        convert_conv_to_gemm = False

    config = get_config(hf_hub)
    hidden_states = Tensor(
        [
            IntVar([batch_size[0], batch_size[1]]),
            IntVar([height[0] // 8, height[1] // 8]),
            IntVar([width[0] // 8, width[1] // 8]),
            config.in_channels,
        ],
        name="hidden_states",
        is_input=True,
    )
    pos_embed = Tensor(
        [1, IntVar([1, 65536]), config.cross_attention_dim],
        name="pos_embed",
        is_input=True,
    )
    encoder_hidden_states = Tensor(
        [1, IntVar([1, 128]), config.caption_channels],
        name="encoder_hidden_states",
        is_input=True,
    )
    timestep = Tensor(
        [IntVar([batch_size[0], batch_size[1]])], name="timestep", is_input=True
    )

    model = Transformer2DModel(
        **config.__dict__,
    )
    model.name_parameter_tensor()

    Y = model.forward(hidden_states, pos_embed, encoder_hidden_states, timestep)
    Y = mark_output(Y, "sample")

    target = detect_target(
        use_fp16_acc=use_fp16_acc, convert_conv_to_gemm=convert_conv_to_gemm
    )
    compile_model(
        Y,
        target,
        work_dir,
        model_name,
        constants=None,
    )


if __name__ == "__main__":
    compile_transformer2d()
