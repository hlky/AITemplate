from typing import Optional, Tuple

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor

from .embeddings import PixArtAlphaCombinedTimestepSizeEmbeddings


class AdaLayerNormSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm single (adaLN-single).

    As proposed in PixArt-Alpha (see: https://arxiv.org/abs/2310.00426; Section 2.3).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        use_additional_conditions (`bool`): To use additional conditions for normalization or not.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_additional_conditions: bool = False,
        dtype: str = "float16",
    ):
        super().__init__()
        self.dtype = dtype
        self.emb = PixArtAlphaCombinedTimestepSizeEmbeddings(
            embedding_dim,
            size_emb_dim=embedding_dim // 3,
            use_additional_conditions=use_additional_conditions,
            dtype=self.dtype,
        )

        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, dtype=dtype)

    def forward(
        self,
        timestep: Tensor,
        resolution: Optional[Tensor] = None,
        aspect_ratio: Optional[Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        # No modulation happening here.
        embedded_timestep = self.emb(timestep, resolution, aspect_ratio, batch_size)
        return self.linear(ops.silu(embedded_timestep)), embedded_timestep
