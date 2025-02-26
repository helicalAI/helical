"""Caduceus config for Hugging Face.

Taken from Caduceus repository: https://github.com/kuleshov-group/caduceus/blob/main/caduceus/configuration_caduceus.py
"""

from typing import Optional, Union

from transformers import PretrainedConfig


class CaduceusPretrainedConfig(PretrainedConfig):
    """Config that extends the original MambaConfig with params relevant to bi-directionality and RC equivariance.

    This config is used to initialise the Caduceus pretrained model.

    Parameters
    ----------
    d_model : int, optional, default=2560
        The model dimension.
    n_layer : int, optional, default=64
        The number of layers.
    vocab_size : int, optional, default=50277
        The vocabulary size.
    ssm_cfg : Optional[dict], optional, default=None
        The configuration for the SSM.
    rms_norm : bool, optional, default=True
        Whether to use RMS normalization.
    residual_in_fp32 : bool, optional, default=True
        Whether to use residual connections in FP32.
    fused_add_norm : bool, optional, default=True
        Whether to use fused add norm.
    pad_vocab_size_multiple : int, optional, default=8
        The padding vocabulary size multiple.
    norm_epsilon : float, optional, default=1e-5
        The epsilon value for layer normalization.
    initializer_cfg : Optional[dict], optional, default=None
        The configuration for the initializer.
    bidirectional : bool, optional, default=True
        Whether to use bidirectional attention.
    bidirectional_strategy : Union[str, None], optional, default="add"
        The bidirectional strategy to use.
    bidirectional_weight_tie : bool, optional, default=True
        Whether to tie the weights for bidirectional attention.
    rcps : bool, optional, default=False
        Whether to use RCPS.
    complement_map : Optional[dict], optional, default=None
        The complement map.
    """

    def __init__(
        self,
        # From original MambaConfig
        d_model: int = 2560,
        n_layer: int = 64,
        vocab_size: int = 50277,
        ssm_cfg: Optional[dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        pad_vocab_size_multiple: int = 8,
        # Not in original MambaConfig, but default arg in create_block in mamba_ssm repo; used in layer norm
        norm_epsilon: float = 1e-5,
        # Used in init_weights
        initializer_cfg: Optional[dict] = None,
        # Caduceus-specific params
        bidirectional: bool = True,
        bidirectional_strategy: Union[str, None] = "add",
        bidirectional_weight_tie: bool = True,
        rcps: bool = False,
        complement_map: Optional[dict] = None,  # used for RCPSEmbedding / RCPSLMHead
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.norm_epsilon = norm_epsilon
        self.initializer_cfg = initializer_cfg
        self.bidirectional = bidirectional
        self.bidirectional_strategy = bidirectional_strategy
        self.bidirectional_weight_tie = bidirectional_weight_tie
        self.rcps = rcps
        self.complement_map = complement_map
