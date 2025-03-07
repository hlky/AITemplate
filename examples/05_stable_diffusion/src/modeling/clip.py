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
from inspect import isfunction
from typing import Optional

from aitemplate.compiler import ops
from aitemplate.frontend import nn, Tensor
from aitemplate.testing import detect_target

# pylint: disable=W0102

USE_CUDA = detect_target().name() == "cuda"


def get_shape(x):
    shape = [it.value() for it in x._attrs["shape"]]
    return shape


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class CrossAttention(nn.Module):
    def __init__(
        self,
        query_dim,
        context_dim=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        dtype="float16",
    ):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False, dtype=dtype)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False, dtype=dtype)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim, dtype=dtype),
            nn.Dropout(dropout, dtype=dtype),
        )

    def forward(self, x, context=None, mask=None, residual=None):
        nheads = self.heads
        d = self.dim_head

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

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
    def __init__(self, dim_in, dim_out, dtype="float16"):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, specialization="mul", dtype=dtype)
        self.gate = nn.Linear(dim_in, dim_out, specialization="fast_gelu", dtype=dtype)

    def forward(self, x):
        return self.proj(x, self.gate(x))


class FeedForward(nn.Module):
    def __init__(
        self, dim, dim_out=None, mult=4, glu=False, dropout=0.0, dtype="float16"
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = (
            nn.Sequential(
                nn.Linear(dim, inner_dim, specialization="fast_gelu", dtype=dtype),
            )
            if not glu
            else GEGLU(dim, inner_dim, dtype=dtype)
        )

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout, dtype=dtype),
            nn.Linear(inner_dim, dim_out, dtype=dtype),
        )

    def forward(self, x, residual=None):
        shape = ops.size()(x)
        x = self.net(x)
        x = ops.reshape()(x, shape)
        if residual is not None:
            return x + residual
        else:
            return x


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        context_dim=None,
        gated_ff=True,
        checkpoint=True,
        only_cross_attention=False,
        dtype="float16",
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.attn1 = CrossAttention(
            query_dim=dim,
            context_dim=context_dim if only_cross_attention else None,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            dtype=dtype,
        )
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff, dtype=dtype)
        if context_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                dtype=dtype,
            )
        else:
            self.attn2 = None
        self.norm1 = nn.LayerNorm(dim, dtype=dtype)
        self.norm2 = nn.LayerNorm(dim, dtype=dtype)
        self.norm3 = nn.LayerNorm(dim, dtype=dtype)
        self.checkpoint = checkpoint

        self.param = (dim, n_heads, d_head, context_dim, gated_ff, checkpoint)

    def forward(self, x, context=None):
        x = self.attn1(
            self.norm1(x),
            residual=x,
            context=context if self.only_cross_attention else None,
        )
        if self.attn2 is not None:
            x = self.attn2(self.norm2(x), context=context, residual=x)
        x = self.ff(self.norm3(x), residual=x)
        return x


def Normalize(in_channels, dtype="float16"):
    return nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True, dtype=dtype
    )


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        depth=1,
        dropout=0.0,
        context_dim=None,
        use_linear_projection=False,
        only_cross_attention=False,
        dtype="float16",
    ):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels, dtype=dtype)  # Group Norm
        self.use_linear_projection = use_linear_projection

        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim, dtype=dtype)
        else:
            self.proj_in = nn.Conv2dBias(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0, dtype=dtype
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    context_dim=context_dim,
                    only_cross_attention=only_cross_attention,
                    dtype=dtype,
                )
                for d in range(depth)
            ]
        )

        if use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels, dtype=dtype)
        else:
            self.proj_out = nn.Conv2dBias(
                inner_dim, in_channels, kernel_size=1, stride=1, padding=0, dtype=dtype
            )

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, h, w, c = x.shape()
        x_in = x
        x = self.norm(x)
        if self.use_linear_projection:
            x = ops.reshape()(x, [b, -1, c])
            x = self.proj_in(x)
        else:
            x = self.proj_in(x)
            x = ops.reshape()(x, [b, -1, c])

        for block in self.transformer_blocks:
            x = block(x, context=context)

        if self.use_linear_projection:
            x = self.proj_out(x)
            x = ops.reshape()(x, [b, h, w, c])
        else:
            x = ops.reshape()(x, [b, h, w, c])
            x = self.proj_out(x)
        return x + x_in


class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        batch_size=1,
        seq_len=16,
        layer_norm_eps=1e-5,
        hidden_dropout_prob=0.0,
        causal=False,
        mask_seq=0,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            dim=hidden_size,
            batch_size=batch_size,
            seq_len=seq_len,
            num_heads=num_attention_heads,
            qkv_bias=True,
            attn_drop=attention_dropout,
            proj_drop=hidden_dropout_prob,
            has_residual=False,
            causal=causal,
            mask_seq=mask_seq,
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = False,
        residual: Optional[Tensor] = None,
    ):
        if residual is not None:
            self_output = self.attn(hidden_states, residual)
        else:
            self_output = self.attn(hidden_states)
        return self_output


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, x):
        x1 = x * 1.702
        x1 = ops.sigmoid(x1)
        x = x * x1
        return x


class CLIPMLP(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer="GELU",
        drop=0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
            specialization="gelu",
        )
        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        shape = x.shape()
        x = self.fc1(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, shape)


class CLIPMLPQuickGelu(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(
            in_features,
            hidden_features,
        )
        self.activation_fn = QuickGELUActivation()

        self.fc2 = nn.Linear(hidden_features, out_features, specialization="add")

    def forward(self, x, res):
        # shape = get_shape(x)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x, res)
        return ops.reshape()(x, x.shape())


class CLIPEncoderLayer(nn.Module):
    ACT_LAYER_TO_CLIP_MLP_MAP = {
        "gelu": CLIPMLP,
        "quick_gelu": CLIPMLPQuickGelu,
    }

    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        attention_dropout=0.0,
        mlp_ratio=4.0,
        batch_size=1,
        seq_len=16,
        causal=False,
        mask_seq=0,
        act_layer="gelu",
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.self_attn = nn.CrossAttention(
            hidden_size,
            seq_len,
            seq_len,
            num_attention_heads,
            qkv_bias=True,
            causal=causal,
        )

        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = self.ACT_LAYER_TO_CLIP_MLP_MAP[act_layer](
            hidden_size, int(hidden_size * mlp_ratio)
        )
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: Tensor,
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, hidden_states, hidden_states, residual
        )

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, residual)

        return hidden_states


class CLIPEncoder(nn.Module):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`CLIPEncoderLayer`].
    Args:
        config: CLIPConfig
    """

    def __init__(
        self,
        num_hidden_layers=12,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        hidden_size=768,
        num_attention_heads=12,
        batch_size=1,
        seq_len=64,
        causal=False,
        mask_seq=0,
        act_layer="gelu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                CLIPEncoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    batch_size=batch_size,
                    seq_len=seq_len,
                    causal=causal,
                    mask_seq=mask_seq,
                    act_layer=act_layer,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict

    def forward(
        self,
        inputs_embeds,
        attention_mask: Optional[Tensor] = None,
        causal_attention_mask: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            causal_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Causal mask for the text model. Mask values selected in `[0, 1]`:
                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
                [What are attention masks?](../glossary#attention-mask)
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.use_return_dict

        encoder_states = () if output_hidden_states else None
        # all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for _, encoder_layer in enumerate(self.layers):
            if output_hidden_states and encoder_states is not None:
                encoder_states = encoder_states + (hidden_states,)
            layer_outputs = encoder_layer(hidden_states)
            hidden_states = layer_outputs

        last_hidden_state = hidden_states
        output = last_hidden_state
        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)
            output = encoder_states
        return output


class CLIPTextEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        vocab_size=49408,
        max_position_embeddings=77,
        dtype="float16",
    ):
        super().__init__()
        embed_dim = hidden_size
        self.max_position_embeddings = max_position_embeddings
        self.embed_dim = hidden_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(shape=[vocab_size, embed_dim], dtype=dtype)
        self.position_embedding = nn.Embedding(
            shape=[max_position_embeddings, embed_dim], dtype=dtype
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Tensor:
        input_shape = ops.size()(input_ids)

        # [B * S]
        token_embedding = self.token_embedding.tensor()
        token_embedding = ops.reshape()(
            token_embedding, [1, self.vocab_size, self.embed_dim]
        )
        token_embedding = ops.expand()(token_embedding, [input_shape[0], -1, -1])

        if inputs_embeds is None:
            inputs_embeds = ops.batch_gather()(token_embedding, input_ids)

        position_embedding = self.position_embedding.tensor()
        position_embedding = ops.reshape()(
            position_embedding, [1, self.max_position_embeddings, self.embed_dim]
        )
        position_embedding = ops.expand()(position_embedding, [input_shape[0], -1, -1])

        position_embeddings = ops.batch_gather()(position_embedding, position_ids)

        embeddings = inputs_embeds + position_embeddings

        embeddings = ops.reshape()(embeddings, [input_shape[0], input_shape[1], -1])

        return embeddings


class CLIPTextTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        text_projection_dim=None,
        output_attentions=False,
        output_hidden_states=False,
        use_return_dict=False,
        num_hidden_layers=12,
        num_attention_heads=12,
        batch_size=1,
        seq_len=64,
        causal=False,
        mask_seq=0,
        act_layer="gelu",
    ):
        super().__init__()
        self.embeddings = CLIPTextEmbeddings(hidden_size=hidden_size)
        self.encoder = CLIPEncoder(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            batch_size=batch_size,
            seq_len=seq_len,
            causal=causal,
            mask_seq=mask_seq,
            act_layer=act_layer,
        )
        self.final_layer_norm = nn.LayerNorm(hidden_size)
        if text_projection_dim is not None:
            self.text_projection = nn.Linear(
                hidden_size, text_projection_dim, bias=False
            )
        else:
            self.text_projection = None

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.use_return_dict = use_return_dict
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_hidden_layers

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        Returns:
        """
        batch = ops.size()(input_ids)[0]

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        encoder_output = self.encoder(
            inputs_embeds=hidden_states, output_hidden_states=self.output_hidden_states
        )
        if self.output_hidden_states:
            last_hidden_state = encoder_output[-1]
        else:
            last_hidden_state = encoder_output
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        argmax = ops.argmax(-1)(input_ids)
        pooled_output = ops.index_select(dim=1)(last_hidden_state, argmax)
        pooled_output = ops.reshape()(pooled_output, [batch, self.hidden_size])
        last_hidden_state._attrs["is_output"] = True
        last_hidden_state._attrs["name"] = "last_hidden_state"
        pooled_output._attrs["is_output"] = True
        pooled_output._attrs["name"] = "pooled_output"
        output = (
            last_hidden_state,
            pooled_output,
        )
        if self.text_projection is not None:
            text_embeds = self.text_projection(pooled_output)
            text_embeds._attrs["is_output"] = True
            text_embeds._attrs["name"] = "text_embeds"
            output = output + (text_embeds,)

        if self.output_hidden_states:
            for idx, hidden_state in enumerate(encoder_output[:-1]):
                hidden_state._attrs["is_output"] = True
                hidden_state._attrs["name"] = f"hidden_state_{idx}"
                output = output + (hidden_state,)

        return output

class CLIPVisionEmbeddings(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_channels=3,
        image_size=224,
        patch_size=16,
        dtype="float16",
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        self.class_embedding = nn.Parameter(shape=[1, 1, hidden_size], dtype=dtype)
        num_channels = num_channels + (4 - (num_channels % 4))
        self.patch_embedding = nn.Conv2dBiasFewChannels(
            in_channels=num_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,
        )

        self.num_patches = (image_size // patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(
            shape=[self.num_positions, hidden_size], dtype=dtype
        )

    def forward(self, pixel_values: Tensor, position_ids: Tensor) -> Tensor:
        pixel_values = ops.pad_last_dim(4, 4)(pixel_values)
        input_shape = ops.size()(pixel_values)
        batch_size = input_shape[0]
        patch_embeds = self.patch_embedding(pixel_values)
        patch_embeds = ops.flatten(1, 2)(patch_embeds)

        class_embeds = ops.expand()(self.class_embedding.tensor(), [batch_size, 1, -1])
        class_embeds._attrs["shape"][0] = pixel_values._attrs["shape"][0]
        embeddings = ops.concatenate()([class_embeds, patch_embeds], dim=1)

        position_embedding = self.position_embedding.tensor()
        position_embedding = ops.reshape()(
            position_embedding, [1, self.num_positions, self.embed_dim]
        )
        position_embedding = ops.expand()(position_embedding, [input_shape[0], -1, -1])

        embeddings = embeddings + ops.batch_gather()(position_embedding, position_ids)

        return embeddings


class CLIPVisionTransformer(nn.Module):
    def __init__(
        self,
        hidden_size=1024,
        layer_norm_eps=1e-05,
        num_channels=3,
        image_size=224,
        patch_size=14,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_act="quick_gelu",
        projection_dim=None,
    ):
        super().__init__()
        self.embed_dim = hidden_size
        self.embeddings = CLIPVisionEmbeddings(
            hidden_size=hidden_size,
            num_channels=num_channels,
            image_size=image_size,
            patch_size=patch_size,
        )
        self.pre_layrnorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.encoder = CLIPEncoder(
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            act_layer=hidden_act,
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        if projection_dim is not None:
            self.visual_projection = nn.Linear(hidden_size, projection_dim, bias=False)
        else:
            self.visual_projection = None

    def forward(
        self,
        pixel_values: Tensor,
        position_ids: Tensor,
    ):
        batch = ops.size()(pixel_values)[0]._attrs["int_var"]._attrs["values"][0]
        hidden_states = self.embeddings(pixel_values, position_ids)
        hidden_states = self.pre_layrnorm(hidden_states)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
        )

        last_hidden_state = encoder_outputs
        pooled_output = ops.dynamic_slice()(
            last_hidden_state,
            start_indices=[0, 0, 0],
            end_indices=[batch, 1, self.embed_dim],
        )
        pooled_output = self.post_layernorm(pooled_output)
        pooled_output = ops.squeeze(dim=0)(pooled_output)
        if self.visual_projection is not None:
            image_embeds = self.visual_projection(pooled_output)
            image_embeds._attrs["is_output"] = True
            image_embeds._attrs["name"] = "image_embeds"
            return image_embeds
        else:
            pooled_output._attrs["is_output"] = True
            pooled_output._attrs["name"] = "pooled_output"
            last_hidden_state._attrs["is_output"] = True
            last_hidden_state._attrs["name"] = "last_hidden_state"
            return pooled_output, last_hidden_state