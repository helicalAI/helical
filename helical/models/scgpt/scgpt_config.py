from typing import Optional, Literal
from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path

class scGPTConfig():
    """
    Configuration class to use the scGPT Model.
    
    Parameters
    ----------
    pad_token : str, optional, default = "<pad>"
        The padding token
    batch_size : int, optional, default = 24
        The batch size
    fast_transformer : bool, optional, default = True
        Whether to use fast transformer or not
    nlayers : int, optional, default = 12
        The number of layers
    nheads : int, optional, default = 8
        The number of heads
    embsize : int, optional, default = 512
        The embedding size
    d_hid : int, optional, default = 512    
        The hidden dimension
    dropout : float, optional, default = 0.2
        The dropout rate
    n_layers_cls : int, optional, default = 3
        The number of classification layers
    mask_value : int, optional, default = -1
        The mask value
    pad_value : int, optional, default = -2
        The padding value
    world_size : int, optional, default = 8
        The world size
    accelerator : bool, optional, default = False
        The accelerator configuration. By default same device as model.
    device : Literal["cpu", "cuda"], optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu".
    use_fast_transformer : bool, optional, default = False
        Wheter to use fast transformer or nots

    Returns
    -------
    scGPTConfig 
       The scGPT configuration object

    Notes
    -----
    This configuration contains all the default parameteres that have been used in the original scGPT repository.

    """

    def __init__(
            self, 
            pad_token: str = "<pad>",
            batch_size: int = 24,
            fast_transformer: bool = True,
            nlayers: int = 12,
            nheads: int = 8,
            embsize: int = 512,
            d_hid: int = 512,
            dropout: float = 0.2,
            n_layers_cls: int = 3,
            mask_value: int = -1,
            pad_value: int = -2,
            world_size: int = 8,
            accelerator: Optional[bool] = False,
            device: Literal["cpu", "cuda"] = "cpu",
            use_fast_transformer: bool = False,
            ):
        
        model_name = 'best_model' # TODO: Include more models
        list_of_files_to_download = [
                                "scgpt/scGPT_CP/vocab.json",
                                f"scgpt/scGPT_CP/{model_name}.pt",
                                ]
        model_path = Path(CACHE_DIR_HELICAL, 'scgpt/scGPT_CP', f'{model_name}.pt')

        self.config = {
            "model_path": model_path,
            "list_of_files_to_download": list_of_files_to_download,
            "pad_token": pad_token,
            "batch_size": batch_size,
            "fast_transformer": fast_transformer,
            "nlayers": nlayers,
            "nheads": nheads,
            "embsize": embsize,
            "d_hid": d_hid,
            "dropout": dropout,
            "n_layers_cls": n_layers_cls,
            "mask_value": mask_value,
            "pad_value": pad_value,
            "world_size": world_size,
            "accelerator": accelerator,
            "device": device,
            "use_fast_transformer": use_fast_transformer,
            }