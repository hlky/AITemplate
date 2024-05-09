from typing import Any, Dict, Optional

from aitemplate.compiler import ops

from aitemplate.frontend import nn, Tensor

from .attention import BasicTransformerBlock
from .embeddings import PatchEmbed, PixArtAlphaTextProjection
from .normalization import AdaLayerNormSingle


class Transformer2DModel(nn.Module):
    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
        use_additional_conditions: Optional[bool] = None,
        dtype: str = "float16",
    ):
        super().__init__()

        # Validate inputs.
        if patch_size is not None:
            if norm_type not in ["ada_norm", "ada_norm_zero", "ada_norm_single"]:
                raise NotImplementedError(
                    f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
                )
            elif (
                norm_type in ["ada_norm", "ada_norm_zero"]
                and num_embeds_ada_norm is None
            ):
                raise ValueError(
                    f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
                )

        self.dtype = dtype
        self.out_channels = in_channels if out_channels is None else out_channels
        self.inner_dim = num_attention_heads * attention_head_dim
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_num_groups = norm_num_groups
        self.cross_attention_dim = cross_attention_dim
        self.attention_bias = attention_bias
        self.sample_size = sample_size
        self.num_vector_embeds = num_vector_embeds
        self.patch_size = patch_size
        self.activation_fn = activation_fn
        self.num_embeds_ada_norm = num_embeds_ada_norm
        self.use_linear_projection = use_linear_projection
        self.only_cross_attention = only_cross_attention
        self.double_self_attention = double_self_attention
        self.upcast_attention = upcast_attention
        self.norm_elementwise_affine = norm_elementwise_affine
        self.norm_eps = norm_eps
        self.attention_type = attention_type
        self.caption_channels = caption_channels
        self.interpolation_scale = interpolation_scale
        self.use_additional_conditions = use_additional_conditions
        if use_additional_conditions is None:
            if norm_type == "ada_norm_single" and sample_size == 128:
                use_additional_conditions = True
            else:
                use_additional_conditions = False
        self.use_additional_conditions = use_additional_conditions

        # 1. Transformer2DModel can process both standard continuous images of shape `(batch_size, num_channels, width, height)` as well as quantized image embeddings of shape `(batch_size, num_image_vectors)`
        # Define whether input is continuous or discrete depending on configuration
        self.is_input_continuous = (in_channels is not None) and (patch_size is None)
        self.is_input_vectorized = num_vector_embeds is not None
        self.is_input_patches = in_channels is not None and patch_size is not None

        if norm_type == "layer_norm" and num_embeds_ada_norm is not None:
            norm_type = "ada_norm"
        self.norm_type = norm_type

        if self.is_input_continuous and self.is_input_vectorized:
            raise ValueError(
                f"Cannot define both `in_channels`: {in_channels} and `num_vector_embeds`: {num_vector_embeds}. Make"
                " sure that either `in_channels` or `num_vector_embeds` is None."
            )
        elif self.is_input_vectorized and self.is_input_patches:
            raise ValueError(
                f"Cannot define both `num_vector_embeds`: {num_vector_embeds} and `patch_size`: {patch_size}. Make"
                " sure that either `num_vector_embeds` or `num_patches` is None."
            )
        elif (
            not self.is_input_continuous
            and not self.is_input_vectorized
            and not self.is_input_patches
        ):
            raise ValueError(
                f"Has to define `in_channels`: {in_channels}, `num_vector_embeds`: {num_vector_embeds}, or patch_size:"
                f" {patch_size}. Make sure that `in_channels`, `num_vector_embeds` or `num_patches` is not None."
            )

        # 2. Initialize the right blocks.
        # These functions follow a common structure:
        # a. Initialize the input blocks. b. Initialize the transformer blocks.
        # c. Initialize the output blocks and other projection blocks when necessary.
        if self.is_input_continuous:
            self._init_continuous_input(norm_type=norm_type)
        elif self.is_input_vectorized:
            self._init_vectorized_inputs(norm_type=norm_type)
        elif self.is_input_patches:
            self._init_patched_inputs(norm_type=norm_type)

    def _init_continuous_input(self, norm_type):
        self.norm = nn.GroupNorm(
            num_groups=self.norm_num_groups,
            num_channels=self.in_channels,
            eps=1e-6,
            affine=True,
            dtype=self.dtype,
        )
        if self.use_linear_projection:
            self.proj_in = nn.Linear(self.in_channels, self.inner_dim, dtype=self.dtype)
        else:
            self.proj_in = nn.Conv2dBias(
                self.in_channels,
                self.inner_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=self.dtype,
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    attention_type=self.attention_type,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.use_linear_projection:
            self.proj_out = nn.Linear(
                self.inner_dim, self.out_channels, dtype=self.dtype
            )
        else:
            self.proj_out = nn.Conv2dBias(
                self.inner_dim,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                dtype=self.dtype,
            )

    def _init_vectorized_inputs(self, norm_type):
        raise NotImplementedError(
            "Vectorized inputs are not yet implemented. `ImagePositionalEmbeddings` is missing."
        )
        assert (
            self.sample_size is not None
        ), "Transformer2DModel over discrete input must provide sample_size"
        assert (
            self.num_vector_embeds is not None
        ), "Transformer2DModel over discrete input must provide num_embed"

        self.height = self.sample_size
        self.width = self.sample_size
        self.num_latent_pixels = self.height * self.width

        self.latent_image_embedding = ImagePositionalEmbeddings(
            num_embed=self.num_vector_embeds,
            embed_dim=self.inner_dim,
            height=self.height,
            width=self.width,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    attention_type=self.attention_type,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.norm_out = nn.LayerNorm(self.inner_dim, dtype=self.dtype)
        self.out = nn.Linear(
            self.inner_dim, self.num_vector_embeds - 1, dtype=self.dtype
        )

    def _init_patched_inputs(self, norm_type):
        assert (
            self.sample_size is not None
        ), "Transformer2DModel over patched input must provide sample_size"

        self.height = self.sample_size
        self.width = self.sample_size

        self.patch_size = self.patch_size
        interpolation_scale = (
            self.interpolation_scale
            if self.interpolation_scale is not None
            else max(self.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed(
            height=self.sample_size,
            width=self.sample_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
            dtype=self.dtype,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.num_attention_heads,
                    self.attention_head_dim,
                    dropout=self.dropout,
                    cross_attention_dim=self.cross_attention_dim,
                    activation_fn=self.activation_fn,
                    num_embeds_ada_norm=self.num_embeds_ada_norm,
                    attention_bias=self.attention_bias,
                    only_cross_attention=self.only_cross_attention,
                    double_self_attention=self.double_self_attention,
                    upcast_attention=self.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.norm_elementwise_affine,
                    norm_eps=self.norm_eps,
                    attention_type=self.attention_type,
                    dtype=self.dtype,
                )
                for _ in range(self.num_layers)
            ]
        )

        if self.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(
                self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=self.dtype
            )
            self.proj_out_1 = nn.Linear(
                self.inner_dim, 2 * self.inner_dim, dtype=self.dtype
            )
            self.proj_out_2 = nn.Linear(
                self.inner_dim,
                self.patch_size * self.patch_size * self.out_channels,
                dtype=self.dtype,
            )
        elif self.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(
                self.inner_dim, elementwise_affine=False, eps=1e-6, dtype=self.dtype
            )
            self.scale_shift_table = nn.Parameter(
                [2, self.inner_dim], dtype=self.dtype, name="scale_shift_table_main"
            )
            self.proj_out = nn.Linear(
                self.inner_dim,
                self.patch_size * self.patch_size * self.out_channels,
                dtype=self.dtype,
            )

        # PixArt-Alpha blocks.
        self.adaln_single = None
        if self.norm_type == "ada_norm_single":
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim,
                use_additional_conditions=self.use_additional_conditions,
                dtype=self.dtype,
            )

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels,
                hidden_size=self.inner_dim,
                dtype=self.dtype,
            )

    def forward(
        self,
        hidden_states: Tensor,
        pos_embed: Tensor,
        encoder_hidden_states: Optional[Tensor] = None,
        timestep: Optional[Tensor] = None,
        added_cond_kwargs: Dict[str, Tensor] = None,
        class_labels: Optional[Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[Tensor] = None,
        encoder_attention_mask: Optional[Tensor] = None,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`LongTensor` of shape `(batch size, num latent pixels)` if discrete, `FloatTensor` of shape `(batch size, height, width, channel)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            raise NotImplementedError(
                "Attention mask is not yet implemented for `Transformer2DModel`."
            )
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            raise NotImplementedError(
                "Encoder attention mask is not yet implemented for `Transformer2DModel`."
            )
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(hidden_states.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch_size, height, width, _ = ops.size()(hidden_states)
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            batch_size, height, width, _ = ops.size()(hidden_states)
            height, width = (
                height._attrs["int_var"] / self.patch_size,
                width._attrs["int_var"] / self.patch_size,
            )
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = (
                self._operate_on_patched_inputs(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    added_cond_kwargs,
                    pos_embed,
                )
            )

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )

        return output

    def _operate_on_continuous_inputs(self, hidden_states: Tensor):
        batch, _, height, width = ops.size()(hidden_states)
        hidden_states = self.norm(hidden_states)

        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = ops.size()(hidden_states, -1)
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(
                batch, height * width, inner_dim
            )
            hidden_states = ops.reshape(
                hidden_states, [batch, height * width, inner_dim]
            )
        else:
            inner_dim = ops.size()(hidden_states, -1)
            hidden_states = ops.reshape(
                hidden_states, [batch, height * width, inner_dim]
            )
            hidden_states = self.proj_in(hidden_states)

        return hidden_states, inner_dim

    def _operate_on_patched_inputs(
        self,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timestep: Tensor,
        added_cond_kwargs,
        pos_embed: Tensor,
    ):
        batch_size = ops.size()(hidden_states, 0)
        hidden_states = self.pos_embed.forward(hidden_states, pos_embed)
        embedded_timestep = None

        if self.adaln_single is not None:
            if self.use_additional_conditions and added_cond_kwargs is None:
                raise ValueError(
                    "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                )
            timestep, embedded_timestep = self.adaln_single(
                timestep, added_cond_kwargs, batch_size=batch_size
            )

        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = ops.reshape()(
                encoder_hidden_states, [batch_size, -1, ops.size()(hidden_states, -1)]
            )

        return hidden_states, encoder_hidden_states, timestep, embedded_timestep

    def _get_output_for_continuous_inputs(
        self,
        hidden_states: Tensor,
        residual: Tensor,
        batch_size,
        height,
        width,
        inner_dim,
    ):
        if not self.use_linear_projection:
            hidden_states = ops.permute()(
                ops.reshape()(hidden_states, [batch_size, height, width, inner_dim]),
                [0, 3, 1, 2],
            )
            # .permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = ops.permute()(
                ops.reshape()(hidden_states, [batch_size, height, width, inner_dim]),
                [0, 3, 1, 2],
            )

        output = hidden_states + residual
        return output

    def _get_output_for_vectorized_inputs(self, hidden_states):
        hidden_states = self.norm_out(hidden_states)
        logits = self.out(hidden_states)
        # (batch, self.num_vector_embeds - 1, self.num_latent_pixels)
        logits = ops.permute021()(logits)
        # log(p(x_0))
        output = ops.softmax()(logits, dim=1)
        output = ops.log()(output)
        return output

    def _get_output_for_patched_inputs(
        self,
        hidden_states,
        timestep,
        class_labels,
        embedded_timestep,
        height=None,
        width=None,
    ):
        batch_size = ops.size()(hidden_states, 0)
        if self.norm_type != "ada_norm_single":
            raise NotImplementedError(
                "`_get_output_for_patched_inputs` is not implemented for `norm_type` other than `ada_norm_single`."
            )
            conditioning = self.transformer_blocks[0].norm1.emb(timestep, class_labels)
            shift, scale = self.proj_out_1(ops.silu(conditioning)).chunk(2, dim=1)
            hidden_states = (
                self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
            )
            hidden_states = self.proj_out_2(hidden_states)
        elif self.norm_type == "ada_norm_single":
            scale_shift_table = ops.unsqueeze(dim=0)(self.scale_shift_table.tensor())
            embedded_timestep = ops.unsqueeze(dim=1)(embedded_timestep)
            shift, scale = ops.chunk()(
                scale_shift_table + embedded_timestep, chunks=2, dim=1
            )
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = hidden_states * (1 + scale) + shift
            hidden_states = self.proj_out(hidden_states)
            hidden_states = ops.squeeze(dim=1)(hidden_states)
        # unpatchify
        if self.adaln_single is None:
            raise NotImplementedError(
                "`_get_output_for_patched_inputs` is not implemented for `norm_type` other than `ada_norm_single`."
            )
            height = width = ops.size()(hidden_states, dim=1) ** 0.5
        hidden_states = ops.reshape()(
            hidden_states,
            [
                batch_size,
                height,
                width,
                self.patch_size,
                self.patch_size,
                self.out_channels,
            ],
        )
        output = ops.reshape()(
            hidden_states,
            [
                batch_size,
                height * self.patch_size,
                width * self.patch_size,
                self.out_channels,
            ],
        )
        return output