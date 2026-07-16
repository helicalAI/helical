"""Self-attention transformer encoder with a LoRA-compatible output projection.

Stock ``torch.nn.MultiheadAttention.forward()`` applies its output projection
via a raw ``F.linear(x, out_proj.weight, out_proj.bias)`` call inside
``F.multi_head_attention_forward`` -- never an actual ``self.out_proj(x)``
module call. PEFT-style LoRA adapters attach by overriding a module's
``forward()``, so an adapter targeting ``out_proj`` never sits on the
gradient path and never trains (helicalAI/bio-agent#1015). ``LoraCompatibleSelfAttention``
below always finishes with an explicit ``self.out_proj(context)`` call, and
``TransformerEncoderLayer``/``TransformerEncoder`` use it in place of stock
``MultiheadAttention``. Shared by scGPT and Nicheformer, which both build on
this same transformer-encoder shape.
"""

import copy
import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import constant_, xavier_uniform_


class LoraCompatibleSelfAttention(nn.Module):
    """Self-attention-only stand-in for ``torch.nn.MultiheadAttention``.

    Supports only self-attention (``query is key is value``): no
    cross-attention, no separate K/V projection weights, no
    ``bias_k``/``bias_v``/``add_zero_attn``. Parameter names
    (``in_proj_weight``, ``in_proj_bias``, ``out_proj.weight``,
    ``out_proj.bias``) match stock ``MultiheadAttention`` exactly, since
    pretrained-checkpoint loading (e.g. scGPT's ``load_pretrained``) matches
    state-dict keys by name and silently skips anything that doesn't match.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first

        self.in_proj_weight = nn.Parameter(
            torch.empty((3 * embed_dim, embed_dim), **factory_kwargs)
        )
        if bias:
            self.in_proj_bias = nn.Parameter(
                torch.empty(3 * embed_dim, **factory_kwargs)
            )
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Matches torch.nn.MultiheadAttention._reset_parameters(): out_proj.weight
        # deliberately keeps nn.Linear's own default (kaiming-uniform) init, since
        # stock MultiheadAttention doesn't xavier-init it either.
        xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        is_causal: bool = False,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if not self.batch_first:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))

        bsz, tgt_len, _ = query.shape
        src_len = key.shape[1]

        merged_mask = self._merge_masks(
            attn_mask, key_padding_mask, query.dtype, src_len
        )
        sdpa_is_causal = False
        if is_causal:
            causal_mask = torch.triu(
                torch.full(
                    (tgt_len, src_len),
                    float("-inf"),
                    device=query.device,
                    dtype=query.dtype,
                ),
                diagonal=1,
            )
            if need_weights or merged_mask is not None:
                merged_mask = (
                    causal_mask if merged_mask is None else merged_mask + causal_mask
                )
            else:
                sdpa_is_causal = True

        qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim).transpose(1, 2)

        if need_weights:
            scale = 1.0 / math.sqrt(self.head_dim)
            scores = (q @ k.transpose(-2, -1)) * scale
            if merged_mask is not None:
                scores = scores + merged_mask
            weights = scores.softmax(dim=-1)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            context = weights @ v
            if average_attn_weights:
                weights = weights.mean(dim=1)
        else:
            context = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=merged_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=sdpa_is_causal,
            )
            weights = None

        context = context.transpose(1, 2).reshape(bsz, tgt_len, self.embed_dim)
        out = self.out_proj(context)  # explicit module call: the actual fix

        if not self.batch_first:
            out = out.transpose(0, 1)

        return out, weights

    @staticmethod
    def _merge_masks(
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        dtype: torch.dtype,
        src_len: int,
    ) -> Optional[Tensor]:
        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=dtype,
        )
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=None,
            other_name="",
            target_type=dtype,
            check_other=False,
        )
        if key_padding_mask is not None:
            # (bsz, src_len) -> (bsz, 1, 1, src_len): additive-bias broadcast over
            # heads and target positions, same merge F.multi_head_attention_forward
            # performs internally.
            kpm = key_padding_mask.reshape(key_padding_mask.shape[0], 1, 1, src_len)
            attn_mask = kpm if attn_mask is None else attn_mask + kpm
        return attn_mask


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


def _get_clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer (self-attn + feedforward), as in `Attention Is
    All You Need <https://arxiv.org/abs/1706.03762>`_, with an ``output_attentions``
    flag threaded through ``forward``/``_sa_block`` and a LoRA-compatible
    ``self_attn`` (see ``LoraCompatibleSelfAttention``).
    """

    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = LoraCompatibleSelfAttention(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.norm2 = nn.LayerNorm(
            d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        output_attentions: bool = False,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )
        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            if output_attentions:
                x, attn_map = x + self._sa_block(
                    self.norm1(x),
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                    output_attentions=output_attentions,
                )
            else:
                x = x + self._sa_block(
                    self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
                )
            x = x + self._ff_block(self.norm2(x))
        else:
            if output_attentions:
                attn_output, attn_map = self._sa_block(
                    x,
                    src_mask,
                    src_key_padding_mask,
                    is_causal=is_causal,
                    output_attentions=output_attentions,
                )
                x = self.norm1(x + attn_output)
            else:
                x = self.norm1(
                    x
                    + self._sa_block(
                        x, src_mask, src_key_padding_mask, is_causal=is_causal
                    )
                )
            x = self.norm2(x + self._ff_block(x))

        if output_attentions:
            return x, attn_map
        return x

    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        output_attentions: bool = False,
    ) -> Tensor:
        if output_attentions:
            x, attn_map = self.self_attn(
                x,
                x,
                x,
                attn_mask=attn_mask,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                is_causal=is_causal,
                average_attn_weights=False,
            )
            return self.dropout1(x), attn_map
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder(nn.Module):
    """Stack of ``N`` ``TransformerEncoderLayer``\\ s, with ``output_attentions``
    plumbed through to collect each layer's attention map.
    """

    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer: TransformerEncoderLayer,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        enable_nested_tensor: bool = True,
        mask_check: bool = True,
    ) -> None:
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # Nested-tensor fast-path optimization is intentionally not implemented
        # here: it's inert during training (self.training is always True during
        # fine-tuning) and this class is used for training, not just inference.
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Tensor:
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype,
        )
        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )
        is_causal = bool(is_causal) if is_causal is not None else False

        output = src
        attn_maps = []
        for mod in self.layers:
            if output_attentions:
                output, attn_map = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                    output_attentions=True,
                )
                attn_maps.append(attn_map)
            else:
                output = mod(
                    output,
                    src_mask=mask,
                    src_key_padding_mask=src_key_padding_mask,
                    is_causal=is_causal,
                )

        if self.norm is not None:
            output = self.norm(output)

        if output_attentions:
            return output, attn_maps
        return output
