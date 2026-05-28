"""
Version of x_transformers.x_transformer to support kv caching for homr's decoder in onnx.
Used x_transformers==2.4.9
I had issues with 2.9.2
"""

# concerning the many noqa statements here:
# I wanted to modify as little as possible of x_transformers code

# F403 and F405 are related to the star import
# S101 is related to the asserts (from x_transformer.py)

# flake8: noqa: S101

from random import random

from torch import Tensor, arange, long, nn
from x_transformers.x_transformers import (  # noqa: F401
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Intermediates,
    LayerIntermediates,
    TokenEmbedding,
    default,
    dropout_seq,
    einx,
    exists,
    first,
    maybe,
    partial,
    rearrange,
    reduce,
    softclamp,
)

__all__ = [
    "AbsolutePositionalEmbedding",
    "AttentionLayers",
    "Intermediates",
    "LayerIntermediates",
    "TokenEmbedding",
]


class CustomAttentionLayers(AttentionLayers):
    def forward(
        self,
        x,
        context=None,
        mask=None,
        context_mask=None,
        attn_mask=None,
        self_attn_kv_mask=None,
        mems=None,
        mem_masks=None,
        seq_start_pos: Tensor | None = None,
        seq_pos_offset: int = 0,
        cache: LayerIntermediates | None = None,
        input_not_include_cache=False,
        cache_age=1,
        return_hiddens=False,
        rotary_pos_emb=None,
        pos=None,
        context_pos=None,
        attn_bias=None,
        deep_embeds_and_ids: tuple[nn.Parameter, Tensor] | None = None,
        condition=None,
        in_attn_cond=None,  # https://arxiv.org/abs/2105.04090
        layers_execute_order: tuple[int, ...] | None = None,
    ):
        assert not (
            self.cross_attend ^ exists(context)
        ), "context must be passed in if cross_attend is set to True"
        assert not (
            exists(condition) ^ self.need_condition
        ), "condition needs to be passed in if using adaptive layernorm or vice versa"

        # handle condition

        if exists(condition):
            assert (
                condition.shape[-1] == self.dim_condition
            ), f"expected condition dimension of {self.dim_condition} but received {condition.shape[-1]}"  # noqa: E501

            assert condition.ndim in {2, 3}

            if condition.ndim == 2:
                condition = rearrange(condition, "b d -> b 1 d")

            condition = self.adaptive_mlp(condition)

        # setup maybe layernorm kwarg

        norm_kwargs = dict()  # noqa: C408

        if self.norm_need_condition:
            norm_kwargs.update(condition=condition)

        # maybe post branch fn conditioning (DiT paper's ada-ln-zero)

        block_forward_kwargs = dict()  # noqa: C408

        if self.post_branch_fn_needs_condition:
            block_forward_kwargs.update(condition=condition)

        # initialize accums

        hiddens = []
        layer_hiddens = []
        intermediates = []

        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        mem_masks = mem_masks.copy() if exists(mem_masks) else [None] * self.num_attn_layers

        # handle left padded sequences

        if exists(seq_start_pos):
            seq_arange = arange(x.shape[-2], device=x.device, dtype=long)
            left_pad_mask = seq_arange >= seq_start_pos[..., None]

            if exists(self_attn_kv_mask):
                self_attn_kv_mask = self_attn_kv_mask & left_pad_mask
            else:
                self_attn_kv_mask = left_pad_mask

        # rotary positions

        cross_attn_rotary_pos_emb = dict()  # noqa: C408

        if exists(self.rotary_pos_emb):
            if not exists(rotary_pos_emb):
                maybe_mem = first(
                    mems, None
                )  # todo - handle edge case where different layers get different memory lengths.
                # don't think this will ever come up but who knows
                mem_len = maybe_mem.shape[1] if exists(maybe_mem) else 0

                if not exists(pos):
                    pos = arange(x.shape[1] + mem_len + seq_pos_offset, device=x.device) - mem_len

                rotary_pos_emb = self.rotary_pos_emb(pos)

            # allow for rotary positions for context if provided

            if exists(context_pos):
                assert self.cross_attend
                context_rotary_pos_emb = self.rotary_pos_emb(context_pos)

                cross_attn_rotary_pos_emb.update(
                    rotary_pos_emb=rotary_pos_emb, context_rotary_pos_emb=context_rotary_pos_emb
                )

        # assume cached key / values

        prev_cache_length = 0

        attn_cache = []
        if exists(cache):
            assert self.causal and not exists(attn_mask)

            prev_cache_length = cache.cache_length

            if exists(context) and False:
                context = context[:, :0]

            if cache_age > 0 and False:
                x = x[:, -cache_age:]  # for spec decoding, may be greater than 1

                if exists(deep_embeds_and_ids):
                    deep_embeds, token_ids = deep_embeds_and_ids
                    token_ids = token_ids[:, -cache_age:]
                    deep_embeds_and_ids = (deep_embeds, token_ids)

            attn_cache = cache.attn_intermediates

        next_cache_length = x.shape[1]

        iter_attn_cache = iter(attn_cache)

        # handle deep embeds if needed

        deep_embeds = []

        if exists(deep_embeds_and_ids):
            deep_embeds, token_ids = deep_embeds_and_ids
            deep_embeds_across_depth = deep_embeds[token_ids]
            deep_embeds = rearrange(deep_embeds_across_depth, "b n l d -> l b n d")

        deep_embeds_iter = iter(deep_embeds)

        # setup multistreams if needed

        streams = self.num_residual_streams
        is_multistream = streams > 1

        if is_multistream:
            x = einx.add("b n d, s d -> (b s) n d", x, self.stream_emb)

        # get layers to be executed

        layer_variables = (
            self.layer_types,
            self.skip_combines,
            self.layers,
            self.layer_dropouts,
            self.layer_integrators,
        )

        # able to override the layers execution order on forward, for trying to depth extrapolate

        layers_execute_order = default(layers_execute_order, self.layers_execute_order)
        layer_variables = tuple(
            tuple(layer_variable[i] for i in layers_execute_order)
            for layer_variable in layer_variables
        )

        # derived input for reinjection if needed

        inp_inject = None

        if self.reinject_input:
            assert not exists(in_attn_cond)
            inp_inject = self.reinject_input_proj(x)

        elif exists(in_attn_cond):
            # handle in-attention conditioning, which serves the same purpose
            # of having the network learn the residual
            inp_inject = (
                in_attn_cond if in_attn_cond.ndim == 3 else rearrange(in_attn_cond, "b d -> b 1 d")
            )

        if exists(inp_inject) and exists(self.learned_reinject_input_gate):
            inp_inject_gate = self.learned_reinject_input_gate(x).sigmoid()
            inp_inject = inp_inject * inp_inject_gate

        # store all hiddens for skips

        skip_hiddens = []

        # for value residuals

        first_self_attn_inter = None
        first_cross_attn_inter = None

        # go through the attention and feedforward layers

        for ind, (
            layer_type,
            skip_combine,
            (norm, block, residual_fn),
            layer_dropout,  # noqa: B905
            layer_integrator,  # noqa: F841
        ) in enumerate(
            zip(*layer_variables)  # noqa: B905
        ):  # noqa: B905
            is_last = ind == (len(self.layers) - 1)  # noqa: F841

            # handle skip connections

            skip_hiddens.append(x)

            if exists(skip_combine):
                x = skip_combine(x, skip_hiddens)

            # layer dropout

            if self.training and layer_dropout > 0.0 and random() < layer_dropout:
                continue

            if layer_type == "a":
                if return_hiddens:
                    hiddens.append(x)

                layer_mem = mems.pop(0) if mems else None
                layer_mem_mask = mem_masks.pop(0) if mem_masks else None

            if layer_type == "c":
                if self.training and self.cross_attn_tokens_dropout > 0.0:
                    context, context_mask = dropout_seq(
                        context, context_mask, self.cross_attn_tokens_dropout
                    )

            x, inner_residual, residual_kwargs = residual_fn.prepare(x)

            layer_hiddens.append(x)

            if exists(layer_integrator):
                x = layer_integrator(x, layer_hiddens)

            pre_norm, post_branch_norm, post_main_norm = norm

            if self.need_condition:
                pre_norm = maybe(partial)(pre_norm, **norm_kwargs)
                post_branch_norm = maybe(partial)(post_branch_norm, **norm_kwargs)
                post_main_norm = maybe(partial)(post_main_norm, **norm_kwargs)

            if exists(inp_inject):
                x = x + inp_inject

            if exists(pre_norm):
                x = pre_norm(x)

                if layer_type == "a" and exists(layer_mem):
                    layer_mem = pre_norm(layer_mem)  # noqa: PLW2901

            block = partial(block, **block_forward_kwargs)  # noqa: PLW2901

            # handle maybe value residuals

            maybe_self_attn_value_residual = None
            maybe_cross_attn_value_residual = None

            if self.add_value_residual:
                if exists(first_self_attn_inter):
                    maybe_self_attn_value_residual = first_self_attn_inter.values

                if exists(first_cross_attn_inter):
                    maybe_cross_attn_value_residual = first_cross_attn_inter.values

            # forward depending on layer type
            layer_cache = None
            if layer_type in ("a", "c"):
                layer_cache = next(iter_attn_cache, None)

            if layer_type == "a":
                out, inter = block(
                    x,
                    mask=mask,
                    context_mask=self_attn_kv_mask,
                    attn_mask=attn_mask,
                    rel_pos=self.rel_pos,
                    pos=pos,
                    rotary_pos_emb=rotary_pos_emb,
                    prev_attn=prev_attn,
                    cache=layer_cache,
                    mem=layer_mem,
                    mem_mask=layer_mem_mask,
                    attn_bias=attn_bias,
                    value_residual=maybe_self_attn_value_residual,
                    return_intermediates=True,
                )
            elif layer_type == "c":
                out, inter = block(
                    x,
                    context=context,
                    mask=mask,
                    context_mask=context_mask,
                    prev_attn=prev_cross_attn,
                    cache=layer_cache,
                    value_residual=maybe_cross_attn_value_residual,
                    **cross_attn_rotary_pos_emb,
                    return_intermediates=True,
                )
            elif layer_type == "f":
                out = block(x, deep_embed=next(deep_embeds_iter, None))

            # store first self or cross attention intermediate for value residual

            if not exists(first_self_attn_inter) and layer_type == "a":
                first_self_attn_inter = inter

            if not exists(first_cross_attn_inter) and layer_type == "c":
                first_cross_attn_inter = inter

            if exists(post_branch_norm):
                out = post_branch_norm(out)

            x = residual_fn(out, inner_residual, **residual_kwargs)

            if layer_type in ("a", "c") and return_hiddens:
                inter.layer_type = layer_type
                intermediates.append(inter)

            if layer_type == "a" and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == "c" and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x)

        if return_hiddens:
            layer_hiddens.append(x)

        if self.softclamp_output:
            x = softclamp(x, self.softclamp_output_value)

        final_norm = self.final_norm

        if self.need_condition:
            final_norm = maybe(partial)(final_norm, **norm_kwargs)

        # take care of multistreams if needed, use sum for now

        if is_multistream:
            x = reduce(x, "(b s) n d -> b n d", "sum", s=streams)

        x = final_norm(x)

        if not return_hiddens:
            return x

        intermediates = LayerIntermediates(
            hiddens=hiddens,
            last_hidden=x,
            attn_intermediates=intermediates,
            layer_hiddens=layer_hiddens,
            cache_length=next_cache_length + prev_cache_length,
        )

        return x, intermediates


class CustomDecoder(CustomAttentionLayers):
    def __init__(self, **kwargs):
        assert "causal" not in kwargs, "cannot set causality on decoder"
        super().__init__(causal=True, **kwargs)
