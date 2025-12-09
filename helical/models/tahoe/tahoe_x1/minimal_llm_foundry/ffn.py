# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Callable
import torch
from copy import deepcopy
import logging
from functools import partial

LOGGER = logging.getLogger(__name__)

_FFN_ACT_FN_DEFAULT = {
    'name': 'gelu',
    'approximate': 'none',
}

def resolve_ffn_hidden_size(
    d_model: int,
    expansion_ratio: Union[int, float],
    ffn_hidden_size: Optional[int] = None,
) -> int:
    """Resolve the hidden size of the feed-forward network.

    Args:
        d_model (int): The dimension of the input and output of the feed-forward network.
        expansion_ratio (Union[int, float]): The expansion ratio of the feed-forward network.
        ffn_hidden_size (Optional[int]): The hidden size of the feed-forward network.

    Returns:
        int: The hidden size of the feed-forward network.
    """
    if ffn_hidden_size is not None:
        LOGGER.info(
            f'`expansion_ratio` (={expansion_ratio}) ignored when `ffn_hidden_size` (={ffn_hidden_size}) is specified.',
        )
    else:
        ffn_hidden_size = int(d_model * expansion_ratio)
        if ffn_hidden_size != d_model * expansion_ratio:
            raise ValueError(
                f'`d_model * expansion_ratio` must be an integer ({d_model=}; {expansion_ratio=}; {d_model * expansion_ratio=}).',
            )
    return ffn_hidden_size

def resolve_ffn_act_fn(
    config: Optional[dict] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Resolve the activation function for the feed-forward network.

    Args:
        config (Optional[dict]): The configuration dictionary for the activation function.
            The dict config must specify the 'name' of a torch.nn.functional activation
            function. All of other key values pairs are bound to the function as a partial.

    Returns:
        Callable[[torch.Tensor], torch.Tensor]: The activation function.
    """
    if config is None:
        config = _FFN_ACT_FN_DEFAULT
    config = deepcopy(config)
    name = config.pop('name')
    if name == 'quick_gelu':
        return quickgelu_activation
    else:
        if not hasattr(torch.nn.functional, name):
            raise ValueError(f'Unrecognized activation function name ({name}).')
        act = getattr(torch.nn.functional, name)
        return partial(act, **config)

def quickgelu_activation(input: torch.Tensor) -> torch.Tensor:
    """Applies GELU approximation that is fast but somewhat inaccurate.

    Args:
        input (torch.Tensor): Input tensor of shape(*), where * means any
            number of dimensions

    Returns:
        torch.Tensor: Tensor with same shape as input tensor
    """
    return input * torch.sigmoid(1.702 * input)