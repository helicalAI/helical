from typing import Optional
from typing import Literal

class UCEConfig():
    """Configuration class to use the Universal Cell-Embedding Model.
    
    Parameters
    ----------
    model_name : Literal["33l_8ep_1024t_1280", "4layer_model"], optional, default = "4layer_model"
        The model name
    batch_size : int, optional, default = 5
        The batch size
    pad_length : int, optional, default = 1536
        The padding length
    pad_token_idx : int, optional, default = 0
        The padding token index
    chrom_token_left_idx : int, optional, default = 1
        The left chrom token index
    chrom_token_right_idx : int, optional, default = 2 
        The right chrom token index 
    cls_token_idx : int, optional, default = 3
        The cls token index
    CHROM_TOKEN_OFFSET : int, optional, default = 143574
        The chrom token offset
    sample_size : int, optional, default = 1024
        The sample size
    CXG : bool, optional, default = True
        Whether to use CXG or not
    n_layers : int, optional, default = 4
        The number of layers. Currently the two pre-trained models available have either 4 layers or 33 layers.
    output_dim : int, optional, default = 1280
        The output dimension
    d_hid : int, optional, default = 5120
        The hidden dimension
    token_dim : int, optional, default = 5120
        The token dimension
    multi_gpu : bool, optional, default = False
        Whether to use multiple GPUs or not
    accelerator : dict, optional, default = {"cpu": True}
        The accelerator configuration

    Returns
    -------
    UCEConfig
        The UCE configuration object
    """
    def __init__(self,
                 model_name: Literal["33l_8ep_1024t_1280", "4layer_model"] = "4layer_model",
                 batch_size: int = 5,
                 pad_length: int = 1536,
                 pad_token_idx: int = 0,
                 chrom_token_left_idx: int = 1,
                 chrom_token_right_idx: int = 2,
                 cls_token_idx: int = 3,
                 CHROM_TOKEN_OFFSET: int = 143574,
                 sample_size: int = 1024,
                 CXG: bool = True,
                 output_dim: int = 1280,
                 d_hid: int = 5120,
                 token_dim: int = 5120,
                 multi_gpu: bool = False,
                 accelerator: Optional[dict] = {"cpu": True}
                ):
        
        # model specific parameters
        self.model_map = {
            "33l_8ep_1024t_1280": {
                'n_layers': 33,
            },
            "4layer_model": {
                'n_layers': 4,
            }
        }

        if model_name not in self.model_map:
            raise ValueError(f"Model name {model_name} not found in available models: {self.model_map.keys()}")

        self.config = {
            "model_name": model_name,
            "batch_size": batch_size,
            "pad_length": pad_length,
            "pad_token_idx": pad_token_idx,
            "chrom_token_left_idx": chrom_token_left_idx,
            "chrom_token_right_idx": chrom_token_right_idx,
            "cls_token_idx": cls_token_idx,
            "CHROM_TOKEN_OFFSET": CHROM_TOKEN_OFFSET,
            "sample_size": sample_size,
            "CXG": CXG,
            "n_layers": self.model_map[model_name]['n_layers'],
            "output_dim": output_dim,
            "d_hid": d_hid,
            "token_dim": token_dim,
            "multi_gpu": multi_gpu,
            "accelerator": accelerator,
        }