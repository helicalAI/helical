"""
Taken from Caduceus repository: https://github.com/kuleshov-group/caduceus/blob/main/caduceus/modeling_caduceus.py
Allows for easier loading without using AutoModel from HuggingFace.
"""

import inspect
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.block import Block  # mambav2 file structure
from torch import nn
from torch.nn import functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention


from mamba_ssm.ops.triton.layer_norm import (
    RMSNorm,
    layer_norm_fn,
    rms_norm_fn,
)  # mambav2 file structure
from .pretrained_config import CaduceusPretrainedConfig
from .modeling_rcps import RCPSAddNormWrapper, RCPSEmbedding, RCPSMambaBlock


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    bidirectional=True,
    bidirectional_strategy="add",
    bidirectional_weight_tie=True,
    rcps=False,
    device=None,
    dtype=None,
):
    """Create Caduceus block.

    Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py
    """
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    bidirectional_kwargs = {
        "bidirectional": bidirectional,
        "bidirectional_strategy": bidirectional_strategy,
        "bidirectional_weight_tie": bidirectional_weight_tie,
    }
    mixer_cls = partial(
        BiMambaWrapper,
        layer_idx=layer_idx,
        **ssm_cfg,
        **bidirectional_kwargs,
        **factory_kwargs,
    )
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )
    block_cls = RCPSMambaBlock if rcps else Block
    # mambav2 compatibility
    if "mlp_cls" in inspect.signature(block_cls.__init__).parameters:
        block = block_cls(
            d_model,
            mixer_cls,
            mlp_cls=nn.Identity,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    else:
        block = block_cls(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block


class BiMambaWrapper(nn.Module):
    """Thin wrapper around Mamba to support bi-directionality."""

    def __init__(
        self,
        d_model: int,
        bidirectional: bool = True,
        bidirectional_strategy: Optional[str] = "add",
        bidirectional_weight_tie: bool = True,
        **mamba_kwargs,
    ):
        super().__init__()
        if bidirectional and bidirectional_strategy is None:
            bidirectional_strategy = "add"  # Default strategy: `add`
        if bidirectional and bidirectional_strategy not in ["add", "ew_multiply"]:
            raise NotImplementedError(
                f"`{bidirectional_strategy}` strategy for bi-directionality is not implemented!"
            )
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.mamba_fwd = Mamba(d_model=d_model, **mamba_kwargs)
        if bidirectional:
            self.mamba_rev = Mamba(d_model=d_model, **mamba_kwargs)
            if (
                bidirectional_weight_tie
            ):  # Tie in and out projections (where most of param count lies)
                self.mamba_rev.in_proj.weight = self.mamba_fwd.in_proj.weight
                self.mamba_rev.in_proj.bias = self.mamba_fwd.in_proj.bias
                self.mamba_rev.out_proj.weight = self.mamba_fwd.out_proj.weight
                self.mamba_rev.out_proj.bias = self.mamba_fwd.out_proj.bias
        else:
            self.mamba_rev = None

    def forward(self, hidden_states, inference_params=None):
        """Bidirectional-enabled forward pass

        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        out = self.mamba_fwd(hidden_states, inference_params=inference_params)
        if self.bidirectional:
            out_rev = self.mamba_rev(
                hidden_states.flip(
                    dims=(1,)
                ),  # Flip along the sequence length dimension
                inference_params=inference_params,
            ).flip(
                dims=(1,)
            )  # Flip back for combining with forward hidden states
            if self.bidirectional_strategy == "add":
                out = out + out_rev
            elif self.bidirectional_strategy == "ew_multiply":
                out = out * out_rev
            else:
                raise NotImplementedError(
                    f"`{self.bidirectional_strategy}` for bi-directionality not implemented!"
                )
        return out


class CaduceusEmbeddings(nn.Module):
    def __init__(
        self,
        config: CaduceusPretrainedConfig,
        device=None,
        dtype=None,
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        if config.rcps:
            self.word_embeddings = RCPSEmbedding(
                config.vocab_size,
                config.d_model,
                config.complement_map,
                **factory_kwargs,
            )
        else:
            self.word_embeddings = nn.Embedding(
                config.vocab_size, config.d_model, **factory_kwargs
            )

    def forward(self, input_ids):
        """
        input_ids: (batch, seqlen)
        """
        return self.word_embeddings(input_ids)


class CaduceusMixerModel(nn.Module):
    def __init__(
        self,
        config: CaduceusPretrainedConfig,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.fused_add_norm = config.fused_add_norm
        self.rcps = config.rcps
        self.residual_in_fp32 = config.residual_in_fp32

        self.embeddings = CaduceusEmbeddings(config, **factory_kwargs)

        # Mamba changes the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        if config.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")

        self.layers = nn.ModuleList(
            [
                create_block(
                    config.d_model,
                    ssm_cfg=config.ssm_cfg,
                    norm_epsilon=config.norm_epsilon,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=i,
                    bidirectional=config.bidirectional,
                    bidirectional_strategy=config.bidirectional_strategy,
                    bidirectional_weight_tie=config.bidirectional_weight_tie,
                    rcps=config.rcps,
                    **factory_kwargs,
                )
                for i in range(config.n_layer)
            ]
        )

        norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model, eps=config.norm_epsilon, **factory_kwargs
        )
        self.norm_f = (
            norm_f
            if (config.fused_add_norm or not config.rcps)
            else RCPSAddNormWrapper(norm_f)
        )

    def forward(self, input_ids, inputs_embeds=None, output_hidden_states=False):
        """Mixer forward."""
        all_hidden_states = []
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embeddings(input_ids)

        residual = None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            # TODO: Add support for gradient checkpointing
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=None
            )

        if not self.fused_add_norm:
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states = self.norm_f(
                    hidden_states, residual=residual, prenorm=False
                )
            else:
                residual = (
                    (hidden_states + residual)
                    if residual is not None
                    else hidden_states
                )
                hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            if self.rcps:
                # Set prenorm=False here since we don't need the residual
                hidden_states_fwd = fused_add_norm_fn(
                    hidden_states[..., : hidden_states.shape[-1] // 2],
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., : hidden_states.shape[-1] // 2],
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states_rc = fused_add_norm_fn(
                    hidden_states[..., hidden_states.shape[-1] // 2 :].flip(
                        dims=[-2, -1]
                    ),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual[..., hidden_states.shape[-1] // 2 :].flip(
                        dims=[-2, -1]
                    ),
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
                hidden_states = torch.cat(
                    [hidden_states_fwd, hidden_states_rc.flip(dims=[-2, -1])], dim=-1
                )
            else:
                # Set prenorm=False here since we don't need the residual
                hidden_states = fused_add_norm_fn(
                    hidden_states,
                    self.norm_f.weight,
                    self.norm_f.bias,
                    eps=self.norm_f.eps,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=self.residual_in_fp32,
                )
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class CaduceusPreTrainedModel(PreTrainedModel):
    """PreTrainedModel wrapper for Caduceus backbone."""

    config_class = CaduceusPretrainedConfig
    base_model_prefix = "caduceus"
    supports_gradient_checkpointing = False
    _no_split_modules = ["BiMambaWrapper"]

    def _init_weights(
        self,
        module,
        initializer_range=0.02,  # Now only used for embedding layer.
        **kwargs,
    ):
        """Adapted from: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/models/mixer_seq_simple.py"""

        n_layer = self.config.n_layer
        initialized_cfg = (
            self.config.initializer_cfg
            if self.config.initializer_cfg is not None
            else {}
        )
        rescale_prenorm_residual = initialized_cfg.get("rescale_prenorm_residual", True)
        initializer_range = initialized_cfg.get("initializer_range", initializer_range)
        n_residuals_per_layer = initialized_cfg.get("n_residuals_per_layer", 1)

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=initializer_range)

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth.
            #   > Scale the weights of residual layers at initialization by a factor of 1/√N where N is the # of
            #   residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["out_proj.weight", "fc2.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(n_residuals_per_layer * n_layer)


class CaduceusModel(CaduceusPreTrainedModel):
    """Caduceus model that can be instantiated using HF patterns."""

    def __init__(
        self, config: CaduceusPretrainedConfig, device=None, dtype=None, **kwargs
    ):
        super().__init__(config)

        if config.rcps:
            assert (
                config.complement_map is not None
            ), "Complement map must be provided for RCPS."

        # Adjust vocab size and complement maps if vocab padding is set.
        if config.vocab_size % config.pad_vocab_size_multiple != 0:
            config.vocab_size += config.pad_vocab_size_multiple - (
                config.vocab_size % config.pad_vocab_size_multiple
            )
        if config.complement_map is not None and config.vocab_size > len(
            config.complement_map
        ):
            for i in range(len(config.complement_map), config.vocab_size):
                config.complement_map[i] = i

        self.config = config
        factory_kwargs = {"device": device, "dtype": dtype}
        self.backbone = CaduceusMixerModel(config, **factory_kwargs, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple, BaseModelOutputWithNoAttention]:
        """HF-compatible forward method."""
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        hidden_states, all_hidden_states = self.backbone(
            input_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        if return_dict:
            return BaseModelOutputWithNoAttention(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
            )
        elif output_hidden_states:
            return hidden_states, all_hidden_states
        else:
            return hidden_states
