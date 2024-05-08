import math

import numpy as np
from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor


def get_2d_sincos_pos_embed(
    embed_dim,
    grid_size,
    cls_token=False,
    extra_tokens=0,
    interpolation_scale=1.0,
    base_size=16,
):
    """
    grid_size: int of the grid height and width return: pos_embed: [grid_size*grid_size, embed_dim] or
    [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)

    grid_h = (
        np.arange(grid_size[0], dtype=np.float32)
        / (grid_size[0] / base_size)
        / interpolation_scale
    )
    grid_w = (
        np.arange(grid_size[1], dtype=np.float32)
        / (grid_size[1] / base_size)
        / interpolation_scale
    )
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position pos: a list of positions to be encoded: size (M,) out: (M, D)
    """
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_timestep_embedding(
    timesteps: Tensor,
    embedding_dim: int,
    flip_sin_to_cos: bool = False,
    downscale_freq_shift: float = 1,
    scale: float = 1,
    max_period: int = 10000,
    dtype: str = "float16",
    arange_name="arange",
):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models: Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param embedding_dim: the dimension of the output. :param max_period: controls the minimum frequency of the
    embeddings. :return: an [N x dim] Tensor of positional embeddings.
    """
    assert timesteps._rank() == 1, "Timesteps should be a 1d-array"

    half_dim = embedding_dim // 2

    exponent = (-math.log(max_period)) * Tensor(
        shape=[half_dim], dtype=dtype, name=arange_name
    )

    exponent = exponent * (1.0 / (half_dim - downscale_freq_shift))

    emb = ops.exp(exponent)
    emb = ops.reshape()(timesteps, [-1, 1]) * ops.reshape()(emb, [1, -1])

    # scale embeddings
    emb = scale * emb

    # concat sine and cosine embeddings
    if flip_sin_to_cos:
        emb = ops.concatenate()(
            [ops.cos(emb), ops.sin(emb)],
            dim=-1,
        )
    else:
        emb = ops.concatenate()(
            [ops.sin(emb), ops.cos(emb)],
            dim=-1,
        )
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        dtype: str = "float16",
    ):
        super().__init__()

        self.linear_1 = nn.Linear(
            in_channels, time_embed_dim, specialization="swish", dtype=dtype
        )
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim, dtype=dtype)

    def forward(self, sample):
        sample = self.linear_1(sample)
        sample = self.linear_2(sample)
        return sample


class Timesteps(nn.Module):
    def __init__(
        self,
        num_channels: int,
        flip_sin_to_cos: bool,
        downscale_freq_shift: float,
        dtype: str = "float16",
        arange_name="arange",
    ):
        super().__init__()
        self.num_channels = num_channels
        self.flip_sin_to_cos = flip_sin_to_cos
        self.downscale_freq_shift = downscale_freq_shift
        self.dtype = dtype
        self.arange_name = arange_name

    def forward(self, timesteps: Tensor):
        t_emb = get_timestep_embedding(
            timesteps,
            self.num_channels,
            flip_sin_to_cos=self.flip_sin_to_cos,
            downscale_freq_shift=self.downscale_freq_shift,
            dtype=self.dtype,
            arange_name=self.arange_name,
        )
        return t_emb


class PixArtAlphaCombinedTimestepSizeEmbeddings(nn.Module):
    """
    For PixArt-Alpha.

    Reference:
    https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L164C9-L168C29
    """

    def __init__(
        self,
        embedding_dim,
        size_emb_dim,
        use_additional_conditions: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.outdim = size_emb_dim
        self.time_proj = Timesteps(
            num_channels=256,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            arange_name="time_proj",
            dtype=dtype,
        )
        self.timestep_embedder = TimestepEmbedding(
            in_channels=256, time_embed_dim=embedding_dim, dtype=dtype
        )

        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions:
            self.additional_condition_proj = Timesteps(
                num_channels=256,
                flip_sin_to_cos=True,
                downscale_freq_shift=0,
                arange_name="additional_condition_proj",
                dtype=dtype,
            )
            self.resolution_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim, dtype=dtype
            )
            self.aspect_ratio_embedder = TimestepEmbedding(
                in_channels=256, time_embed_dim=size_emb_dim, dtype=dtype
            )

    def forward(
        self, timestep: Tensor, resolution: Tensor, aspect_ratio: Tensor, batch_size
    ):
        timesteps_proj = self.time_proj(timestep)
        timesteps_emb = self.timestep_embedder(timesteps_proj)  # (N, D)

        if self.use_additional_conditions:
            assert (
                resolution is not None and aspect_ratio is not None
            ), "Additional conditions are required."
            resolution_emb = self.additional_condition_proj(ops.flatten()(resolution))
            resolution_emb = ops.reshape()(
                self.resolution_embedder(resolution_emb), [batch_size, -1]
            )
            aspect_ratio_emb = self.additional_condition_proj(
                ops.flatten()(aspect_ratio)
            )
            aspect_ratio_emb = ops.reshape()(
                self.aspect_ratio_embedder(aspect_ratio_emb), [batch_size, -1]
            )
            conditioning = timesteps_emb + ops.concatenate()(
                [resolution_emb, aspect_ratio_emb], dim=1
            )
        else:
            conditioning = timesteps_emb

        return conditioning


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(
        self,
        height: int = 224,
        width: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        layer_norm: bool = False,
        flatten: bool = True,
        bias: bool = True,
        interpolation_scale: float = 1,
        dtype: str = "float16",
    ):
        super().__init__()

        self.dtype = dtype
        self.flatten = flatten
        self.layer_norm = layer_norm

        if bias:
            self.proj = nn.Conv2dBias(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                dtype=dtype,
            )
        else:
            self.proj = nn.Conv2d(
                in_channels,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
                dtype=dtype,
            )
        if layer_norm:
            self.norm = nn.LayerNorm(
                embed_dim, elementwise_affine=False, eps=1e-6, dtype=dtype
            )
        else:
            self.norm = None

        self.patch_size = patch_size
        # See:
        # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L161
        self.height, self.width = height // patch_size, width // patch_size
        self.base_size = height // patch_size
        self.interpolation_scale = interpolation_scale

    def forward(self, latent: Tensor, pos_embed: Tensor):
        latent = self.proj(latent)
        if self.flatten:
            latent = ops.flatten(1, 2)(latent)
        if self.layer_norm:
            latent = self.norm(latent)

        pos_embed._attrs["shape"][1] = latent._attrs["shape"][1]
        return latent + pos_embed


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(
        self, in_features, hidden_size, num_tokens=120, dtype: str = "float16"
    ):
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features, hidden_size, specialization="fast_gelu", dtype=dtype
        )
        self.linear_2 = nn.Linear(hidden_size, hidden_size, dtype=dtype)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states
