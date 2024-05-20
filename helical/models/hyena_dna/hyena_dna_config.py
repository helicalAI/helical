from typing import Literal
class HyenaDNAConfig():
    """
    Configuration class for Hyena DNA model.

    Args:
        model_name (Literal["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"], optional):
            The name of the model. Defaults to "hyenadna-tiny-1k-seqlen".
        n_layer (int, optional): The number of layers in the model. Defaults to 2.
        vocab_size (int, optional): The size of the vocabulary. Defaults to 12.
        resid_dropout (float, optional): The dropout rate for residual connections. Defaults to 0.0.
        embed_dropout (float, optional): The dropout rate for embedding layer. Defaults to 0.1.
        fused_mlp (bool, optional): Whether to use fused MLP. Defaults to False.
        fused_dropout_add_ln (bool, optional): Whether to use fused dropout and layer normalization. Defaults to True.
        residual_in_fp32 (bool, optional): Whether to use FP32 for residual connections. Defaults to True.
        pad_vocab_size_multiple (int, optional): The multiple to pad the vocabulary size. Defaults to 8.
        return_hidden_state (bool, optional): Whether to return the hidden state. Defaults to True.
        layer (dict, optional): Dictionary containing layer-specific parameters. Defaults to {
            "_name_": "hyena",
            "emb_dim": 5,
            "filter_order": 64,
            "local_order": 3,
            "l_max": 1026,
            "modulate": True,
            "w": 10,
            "lr": 6e-4,
            "wd": 0.0,
            "lr_pos_emb": 0.0
        }.

    Attributes:
        model_map (dict): A dictionary mapping model names to their corresponding configuration parameters.
        config (dict): A dictionary containing the configuration parameters for the Hyena DNA model.

    Raises:
        ValueError: If the specified model name is not found in the available models.

    """

    def __init__(
            self, 
            model_name: Literal["hyenadna-tiny-1k-seqlen", "hyenadna-tiny-1k-seqlen-d256"] = "hyenadna-tiny-1k-seqlen",
            n_layer: int = 2,
            vocab_size: int = 12,
            resid_dropout: float = 0.0,
            embed_dropout: float = 0.1,
            fused_mlp: bool = False,   
            fused_dropout_add_ln: bool = True,
            residual_in_fp32: bool = True,
            pad_vocab_size_multiple: int = 8,
            return_hidden_state: bool = True,
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
                "lr_pos_emb": 0.0
            }
        ):
        
        # model specific parameters
        self.model_map = {
            "hyenadna-tiny-1k-seqlen": {
                'd_model': 128,
                'd_inner': 512,
            },
            "hyenadna-tiny-1k-seqlen-d256": {
                'd_model': 256,
                'd_inner': 1024,
            }
        }

        if model_name not in self.model_map:
            raise ValueError(f"Model name {model_name} not found in available models: {self.model_map.keys()}")

        self.config = {
            "model_name": model_name,
            "d_model": self.model_map[model_name]['d_model'],
            "n_layer": n_layer,
            "d_inner": self.model_map[model_name]['d_inner'],
            "vocab_size": vocab_size,
            "resid_dropout": resid_dropout,
            "embed_dropout": embed_dropout,
            "fused_mlp": fused_mlp,
            "fused_dropout_add_ln": fused_dropout_add_ln,
            "residual_in_fp32": residual_in_fp32,
            "pad_vocab_size_multiple": pad_vocab_size_multiple,
            "return_hidden_state": return_hidden_state,
            "layer": layer
        }



