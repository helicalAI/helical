from typing import Optional
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal


class GenePTConfig:
    """Configuration class to use the GenePT Model.

    Parameters
    ----------
    model_name : Literal["gpt3.5"], optional, default="gpt3.5"
        The name of the model for the embeddings.
    batch_size : int, optional, default = 24
        The batch size
    emb_layer : int, optional, default = -1
        The embedding layer
    emb_mode : Literal["cls", "cell", "gene"], optional, default="cell"
        The embedding mode
    device : Literal["cpu", "cuda"], optional, default="cpu"
        The device to use. Either use "cuda" or "cpu".
    accelerator : bool, optional, default=False
        The accelerator configuration. By default same device as model.
    nproc: int, optional, default=1
        Number of processes to use for data processing.
    custom_attr_name_dict : dict, optional, default=None
        A dictionary that contains the names of the custom attributes to be added to the dataset.
        The keys of the dictionary are the names of the custom attributes, and the values are the names of the columns in adata.obs.
        For example, if you want to add a custom attribute called "cell_type" to the dataset, you would pass custom_attr_name_dict = {"cell_type": "cell_type"}.
        If you do not want to add any custom attributes, you can leave this parameter as None.
    Returns
    -------
    GenePTConfig
        The GenePT configuration object

    """

    def __init__(
        self,
        model_name: Literal["gpt3.5"] = "gpt3.5",
        batch_size: int = 24,
        emb_layer: int = -1,
        emb_mode: Literal["cls", "cell", "gene"] = "cell",
        device: Literal["cpu", "cuda"] = "cpu",
        accelerator: Optional[bool] = False,
        nproc: int = 1,
        custom_attr_name_dict: Optional[dict] = None,
    ):

        # model specific parameters
        self.model_map = {
            "gpt3.5": {
                "input_size": 4096,
                "special_token": True,
                "embsize": 512,
            }
        }
        if model_name not in self.model_map:
            raise ValueError(
                f"Model name {model_name} not found in available models: {self.model_map.keys()}"
            )
        list_of_files_to_download = [
            "genept/genept_embeddings/genept_embeddings.json",
        ]

        embeddings_path = Path(
            CACHE_DIR_HELICAL, "genept/genept_embeddings/genept_embeddings.json"
        )

        self.config = {
            "embeddings_path": embeddings_path,
            "model_name": model_name,
            "batch_size": batch_size,
            "emb_layer": emb_layer,
            "emb_mode": emb_mode,
            "device": device,
            "accelerator": accelerator,
            "input_size": self.model_map[model_name]["input_size"],
            "special_token": self.model_map[model_name]["special_token"],
            "embsize": self.model_map[model_name]["embsize"],
            "nproc": nproc,
            "custom_attr_name_dict": custom_attr_name_dict,
            "list_of_files_to_download": list_of_files_to_download,
        }
