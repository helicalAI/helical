from typing import Optional
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal

class GeneformerConfig():
    """Configuration class to use the Geneformer Model.
    
    Parameters
    ----------
    embed_obsm_name : str, optional, default = "X_geneformer"
        The name of the obsm under which the embeddings will be saved in the AnnData object
    batch_size : int, optional, default = 5
        The batch size
    emb_layer : int, optional, default = -1
        The embedding layer
    emb_mode : str, optional, default = "cell"
        The embedding mode
    device : Literal["cpu", "cuda"], optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu".
    accelerator : bool, optional, default = False
        The accelerator configuration. By default same device as model.

    Returns
    -------
    GeneformerConfig
        The Geneformer configuration object

    """
    def __init__(
            self, 
            embed_obsm_name: str = "X_geneformer",
            batch_size: int = 5,
            emb_layer: int = -1,
            emb_mode: str = "cell",
            device: Literal["cpu", "cuda"] = "cpu",
            accelerator: Optional[bool] = False
            ):
        
        model_name = "geneformer-12L-30M"
        
        self.list_of_files_to_download = [
            "geneformer/gene_median_dictionary.pkl",
            "geneformer/human_gene_to_ensemble_id.pkl",
            "geneformer/token_dictionary.pkl",
            f"geneformer/{model_name}/config.json",
            f"geneformer/{model_name}/pytorch_model.bin",
            f"geneformer/{model_name}/training_args.bin",
            ]

        self.model_dir = Path(CACHE_DIR_HELICAL, 'geneformer')
        self.model_name = model_name
        self.embed_obsm_name = embed_obsm_name
        self.batch_size = batch_size
        self.emb_layer = emb_layer
        self.emb_mode = emb_mode
        self.device = device
        self.accelerator = accelerator


