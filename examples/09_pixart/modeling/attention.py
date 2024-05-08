from inspect import isfunction
from typing import Any, Dict, Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Attention(nn.Module):
    def __init__(
        self,
        query_dim,
        heads=8,
        dim_head=64,
        cross_attention_dim=None,
        dropout=0.0,
        bias=True,
        out_bias=True,
        dtype="float16",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        cross_attention_dim = default(cross_attention_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=bias, dtype=dtype)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype, bias=out_bias),
            nn.Dropout(dropout, dtype=dtype),
        )

    def forward(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        attention_mask=None,
        residual=None,
    ):
        nheads = self.heads
        d = self.dim_head

        q = self.to_q(hidden_states)
        encoder_hidden_states = default(encoder_hidden_states, hidden_states)
        k = self.to_k(encoder_hidden_states)
        v = self.to_v(encoder_hidden_states)

        bs = q.shape()[0]

        q = ops.reshape()(q, [bs, -1, self.heads, self.dim_head])
        k = ops.reshape()(k, [bs, -1, self.heads, self.dim_head])
        v = ops.reshape()(v, [bs, -1, self.heads, self.dim_head])
        q = ops.permute()(q, [0, 2, 1, 3])
        k = ops.permute()(k, [0, 2, 1, 3])
        v = ops.permute()(v, [0, 2, 1, 3])

        attn_op = ops.mem_eff_attention(causal=False)
        out = attn_op(
            (ops.reshape()(q, [bs, nheads, -1, d])),
            (ops.reshape()(k, [bs, nheads, -1, d])),
            (ops.reshape()(v, [bs, nheads, -1, d])),
        )
        out = ops.reshape()(out, [bs, -1, nheads * d])
        proj = self.to_out(out)
        proj = ops.reshape()(proj, [bs, -1, nheads * d])
        if residual is not None:
            return proj + residual
        else:
            return proj


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, dtype="float16"):
        super().__init__()
        self.proj = nn.Linear(
            dim_in, dim_out, bias=bias, specialization="mul", dtype=dtype
        )
        self.gate = nn.Linear(
            dim_in, dim_out, bias=bias, specialization="fast_gelu", dtype=dtype
        )

    def forward(self, x):
        return self.proj(x, self.gate(x))


class ApproximateGELU(nn.Module):
    r"""
    The approximate form of the Gaussian Error Linear Unit (GELU). For more details, see section 2 of this
    [paper](https://arxiv.org/abs/1606.08415).

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(self, dim_in: int, dim_out: int, bias: bool = True):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        return x * ops.sigmoid(1.702 * x)


class GELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        specialization: str = "gelu",
        bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.proj = nn.Linear(
            dim_in, dim_out, bias=bias, specialization=specialization, dtype=dtype
        )

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        return hidden_states


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        dtype="float16",
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, specialization="gelu", dtype=dtype)
        elif activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, specialization="fast_gelu", dtype=dtype)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, dtype=dtype)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList(
            [
                act_fn,
                nn.Dropout(dropout, dtype=dtype),
                nn.Linear(inner_dim, dim_out, dtype=dtype),
            ]
        )
        if final_dropout:
            self.net.append(nn.Dropout(dropout, dtype=dtype))

    def forward(self, hidden_states: Tensor) -> Tensor:
        for layer in self.net:
            hidden_states = layer(hidden_states)
        return hidden_states


class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        dtype: str = "float16",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (
            num_embeds_ada_norm is not None
        ) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            raise NotImplementedError(
                "`sinusoidal` positional embeddings are not implemented yet."
            )
            self.pos_embed = SinusoidalPositionalEmbedding(
                dim, max_seq_length=num_positional_embeddings
            )
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            raise NotImplementedError("`ada_norm` is not implemented yet.")
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            raise NotImplementedError("`ada_norm_zero` is not implemented yet.")
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            raise NotImplementedError("`ada_norm_continuous` is not implemented yet.")
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            self.norm1 = nn.LayerNorm(
                dim,
                elementwise_affine=norm_elementwise_affine,
                eps=norm_eps,
                dtype=dtype,
            )

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            out_bias=attention_out_bias,
            dtype=dtype,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                raise NotImplementedError("`ada_norm` is not implemented yet.")
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                raise NotImplementedError(
                    "`ada_norm_continuous` is not implemented yet."
                )
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                self.norm2 = nn.LayerNorm(
                    dim,
                    norm_eps,
                    elementwise_affine=norm_elementwise_affine,
                    dtype=dtype,
                )

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=(
                    cross_attention_dim if not double_self_attention else None
                ),
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                out_bias=attention_out_bias,
                dtype=dtype,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            raise NotImplementedError("`ada_norm_continuous` is not implemented yet.")
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in [
            "ada_norm_zero",
            "ada_norm",
            "layer_norm",
            "ada_norm_continuous",
        ]:
            self.norm3 = nn.LayerNorm(
                dim, norm_eps, elementwise_affine=norm_elementwise_affine, dtype=dtype
            )
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
            dtype=dtype,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            raise NotImplementedError(
                "`gated` and `gated-text-image` attention types are not implemented yet."
            )
            self.fuser = GatedSelfAttentionDense(
                dim, cross_attention_dim, num_attention_heads, attention_head_dim
            )

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter([6, dim], dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        encoder_hidden_states: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, Tensor]] = None,
    ) -> Tensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = ops.size()(hidden_states, dim=0)

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif self.norm_type == "ada_norm_single":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = ops.chunk()(
                ops.unsqueeze(dim=0)(self.scale_shift_table.tensor())
                + ops.reshape()(timestep, [batch_size, 6, -1]),
                chunks=6,
                dim=1,
            )
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = ops.squeeze(dim=1)(norm_hidden_states)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        # cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        # gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=(
                encoder_hidden_states if self.only_cross_attention else None
            ),
            attention_mask=attention_mask,
            # **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            raise NotImplementedError("`ada_norm_zero` is not implemented yet.")
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states._rank() == 4:
            hidden_states = ops.squeeze(dim=1)(hidden_states)

        # 1.2 GLIGEN Control
        # if gligen_kwargs is not None:
        #     hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                raise NotImplementedError("`ada_norm` is not implemented yet.")
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                raise NotImplementedError(
                    "`ada_norm_continuous` is not implemented yet."
                )
                norm_hidden_states = self.norm2(
                    hidden_states, added_cond_kwargs["pooled_text_emb"]
                )
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                raise NotImplementedError("`pos_embed` is not implemented yet.")
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            raise NotImplementedError("`ada_norm_continuous` is not implemented yet.")
            norm_hidden_states = self.norm3(
                hidden_states, added_cond_kwargs["pooled_text_emb"]
            )
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            raise NotImplementedError("`ada_norm_zero` is not implemented yet.")
            norm_hidden_states = (
                norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            )

        if self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            raise NotImplementedError("`ada_norm_zero` is not implemented yet.")
            ff_output = ops.unsqueeze(dim=1)(gate_mlp) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states._rank() == 4:
            hidden_states = ops.squeeze(dim=1)(hidden_states)

        return hidden_states
