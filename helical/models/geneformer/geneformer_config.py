from typing import Optional
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal

class GeneformerConfig():
    """Configuration class to use the Geneformer Model.
    
    Parameters
    ----------
    model_name : Literal["gf-6L-30M-i2048", "gf-12L-30M-i2048", "gf-12L-95M-i4096", "gf-20L-95M-i4096", "gf-12L-95M-i4096-CLcancer"], optional, default = "gf-12L-30M-i2048"
        The name of the model.
    batch_size : int, optional, default = 24
        The batch size
    emb_layer : int, optional, default = -1
        The embedding layer
    emb_mode : str, optional, default = "cell"
        The embedding mode
    device : Literal["cpu", "cuda"], optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu".
    accelerator : bool, optional, default = False
        The accelerator configuration. By default same device as model.
    nproc: int, optional, default = 1
        Number of processes to use for data processing.
    custom_attr_name_dict : dict, optional, default = None
        A dictionary that contains the names of the custom attributes to be added to the dataset. 
        The keys of the dictionary are the names of the custom attributes, and the values are the names of the columns in adata.obs. 
        For example, if you want to add a custom attribute called "cell_type" to the dataset, you would pass custom_attr_name_dict = {"cell_type": "cell_type"}.
        If you do not want to add any custom attributes, you can leave this parameter as None.

    Returns
    -------
    GeneformerConfig
        The Geneformer configuration object

    """
    def __init__(
            self, 
            model_name: Literal["gf-6L-30M-i2048", "gf-12L-30M-i2048", "gf-12L-95M-i4096", "gf-20L-95M-i4096", "gf-12L-95M-i4096-CLcancer"] = "gf-12L-30M-i2048",
            batch_size: int = 24,
            emb_layer: int = -1,
            emb_mode: Literal["cls", "cell", "gene"] = "cell",
            device: Literal["cpu", "cuda"] = "cpu",
            accelerator: Optional[bool] = False,
            nproc: int = 1,
            custom_attr_name_dict: Optional[dict] = None
            ):
        
        # model specific parameters
        self.model_map = {
            "gf-12L-95M-i4096": {
                'input_size': 4096,
                'special_token': True,
                'embsize': 512,
            },
            "gf-12L-95M-i4096-CLcancer": {
                'input_size': 4096,
                'special_token': True,
                'embsize': 512,
            },
            "gf-20L-95M-i4096": {
                'input_size': 4096,
                'special_token': True,
                'embsize': 896,
            },
            "gf-12L-30M-i2048": {
                'input_size': 2048,
                'special_token': False,
                'embsize': 512,
            },
            "gf-6L-30M-i2048": {
                'input_size': 2048,
                'special_token': False,
                'embsize': 256,
            },

        }
        if model_name not in self.model_map:
            raise ValueError(f"Model name {model_name} not found in available models: {self.model_map.keys()}")
        
        model_version = 'v2' if "95M" in model_name else 'v1'

        if model_version == 'v1' and emb_mode=='cls':
            raise ValueError(f"Verson 1 Geneformer models do not support the CLS embedding mode")
        
        self.list_of_files_to_download = [
            f"geneformer/{model_version}/gene_median_dictionary.pkl",
            f"geneformer/{model_version}/token_dictionary.pkl",
            f"geneformer/{model_version}/ensembl_mapping_dict.pkl",
            f"geneformer/{model_version}/{model_name}/config.json",
            f"geneformer/{model_version}/{model_name}/training_args.bin",
        ]

        # Add model weight files to download based on the model version (v1 or v2)
        if model_version == 'v2':
            self.list_of_files_to_download.append(f"geneformer/{model_version}/{model_name}/generation_config.json")
            self.list_of_files_to_download.append(f"geneformer/{model_version}/{model_name}/model.safetensors")
        else:
            self.list_of_files_to_download.append(f"geneformer/{model_version}/{model_name}/pytorch_model.bin")

        self.model_dir = Path(CACHE_DIR_HELICAL, 'geneformer')

        self.files_config = {
            "model_files_dir": Path(CACHE_DIR_HELICAL, 'geneformer', model_version, model_name),
            "gene_median_path": self.model_dir / model_version / "gene_median_dictionary.pkl",
            "token_path": self.model_dir / model_version / "token_dictionary.pkl",
            "ensembl_dict_path": self.model_dir / model_version / "ensembl_mapping_dict.pkl",
        }

        self.config = {
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
            "custom_attr_name_dict": custom_attr_name_dict
        }
    


