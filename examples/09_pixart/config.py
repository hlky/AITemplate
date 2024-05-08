from dataclasses import dataclass

import requests

config_url = "https://huggingface.co/{hf_hub}/raw/main/transformer/config.json"


@dataclass(match_args=False)
class Transformer2DModelConfig:
    activation_fn: str = "gelu-approximate"
    attention_bias: bool = True
    attention_head_dim: int = 72
    attention_type: str = "default"
    caption_channels: int = 4096
    cross_attention_dim: int = 1152
    double_self_attention: bool = False
    dropout: float = 0.0
    in_channels: int = 4
    interpolation_scale: int = 2
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-06
    norm_num_groups: int = 32
    norm_type: str = "ada_norm_single"
    num_attention_heads: int = 16
    num_embeds_ada_norm: int = 1000
    num_layers: int = 28
    num_vector_embeds: int = None
    only_cross_attention: bool = False
    out_channels: int = 8
    patch_size: int = 2
    sample_size: int = 128
    upcast_attention: bool = False
    use_additional_conditions: bool = False
    use_linear_projection: bool = False


def get_config(hf_hub):
    response = requests.get(config_url.format(hf_hub=hf_hub))
    response.raise_for_status()
    data = response.json()
    data = {
        k: v for k, v in data.items() if k in Transformer2DModelConfig.__dict__.keys()
    }
    return Transformer2DModelConfig(**data)
