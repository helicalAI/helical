from typing import Literal
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL


class HyenaDNAConfig:
    """
    Configuration class for Hyena DNA model.

    Parameters
    ----------
    model_name : Literal["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"], optional, default="hyenadna-tiny-1k-seqlen"
        The name of the model.
    batch_size : int, optional, default=5
        The batch size to use for all tasks.
    n_layer : int, optional, default=2
        The number of layers in the model.
    vocab_size : int, optional, default=12
        The size of the vocabulary.
    resid_dropout : float, optional, default=0.0
        The dropout rate for residual connections.
    embed_dropout : float, optional, default=0.1
        The dropout rate for embedding layer.
    fused_mlp : bool, optional, default=False
        Whether to use fused MLP.
    fused_dropout_add_ln : bool, optional, default=True
        Whether to use fused dropout and layer normalization.
    residual_in_fp32 : bool, optional, default=True
        Whether to use FP32 for residual connections.
    checkpoint_mixer : bool, optional, default=False
        Whether to use checkpointing for mixer layers.
    checkpoint_mlp : bool, optional, default=False
        Whether to use checkpointing for MLP layers.
    pad_vocab_size_multiple : int, optional, default=8
        The multiple to pad the vocabulary size.
    return_hidden_state : bool, optional, default=True
        Whether to return the hidden state.
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use.
    layer : dict, optional, default={"_name_": "hyena", "emb_dim": 5, "filter_order": 64, "local_order": 3, "l_max": 1026, "modulate": True, "w": 10, "lr": 6e-4, "wd": 0.0, "lr_pos_emb": 0.0}
        Dictionary containing layer-specific parameters.

    Attributes
    ----------
    model_map : dict
        A dictionary mapping model names to their corresponding configuration parameters.
    config : dict
        A dictionary containing the configuration parameters for the Hyena DNA model.

    Raises
    ------
    ValueError
        If the specified model name is not found in the available models.

    """

    def __init__(
        self,
        model_name: Literal[
            "hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"
        ] = "hyenadna-tiny-1k-seqlen",
        batch_size: int = 5,
        n_layer: int = 2,
        vocab_size: int = 12,
        resid_dropout: float = 0.0,
        embed_dropout: float = 0.1,
        fused_mlp: bool = False,
        fused_dropout_add_ln: bool = True,
        residual_in_fp32: bool = True,
        checkpoint_mixer: bool = False,
        checkpoint_mlp: bool = False,
        pad_vocab_size_multiple: int = 8,
        return_hidden_state: bool = True,
        device: Literal["cpu", "cuda"] = "cpu",
        layer: dict = {
            "_name_": "hyena",
            "emb_dim": 5,
            "filter_order": 64,
            "local_order": 3,
            "l_max": 1026,
            "modulate": True,
            "w": 10,
            "lr": 6e-4,
            "wd": 0.0,
            "lr_pos_emb": 0.0,
        },
    ):

        # model specific parameters
        self.model_map = {
            "hyenadna-tiny-1k-seqlen": {
                "d_model": 128,
                "d_inner": 512,
                "max_length": 1024,  # for max_length see https://github.com/HazyResearch/hyena-dna/blob/main/huggingface.py
            },
            "hyenadna-tiny-1k-seqlen-d256": {
                "d_model": 256,
                "d_inner": 1024,
                "max_length": 1024,
            },
        }

        if model_name not in self.model_map:
            raise ValueError(
                f"Model name {model_name} not found in available models: {self.model_map.keys()}"
            )

        list_of_files_to_download = [f"hyena_dna/{model_name}.ckpt"]

        self.config = {
            "model_name": model_name,
            "model_path": Path(CACHE_DIR_HELICAL, f"hyena_dna/{model_name}.ckpt"),
            "list_of_files_to_download": list_of_files_to_download,
            "batch_size": batch_size,
            "d_model": self.model_map[model_name]["d_model"],
            "n_layer": n_layer,
            "d_inner": self.model_map[model_name]["d_inner"],
            "vocab_size": vocab_size,
            "resid_dropout": resid_dropout,
            "embed_dropout": embed_dropout,
            "fused_mlp": fused_mlp,
            "fused_dropout_add_ln": fused_dropout_add_ln,
            "residual_in_fp32": residual_in_fp32,
            "checkpoint_mixer": checkpoint_mixer,
            "checkpoint_mlp": checkpoint_mlp,
            "pad_vocab_size_multiple": pad_vocab_size_multiple,
            "return_hidden_state": return_hidden_state,
            "device": device,
            "layer": layer,
            "max_length": self.model_map[model_name]["max_length"],
        }
