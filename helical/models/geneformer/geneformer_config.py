from typing import Optional
from helical.services.downloader import Downloader
from pathlib import Path
import os

class GeneformerConfig():
    """Configuration class to use the Geneformer Model.
    
    Parameters
    ----------
    batch_size : int, optional, default = 5
        The batch size
    emb_layer : int, optional, default = -1
        The embedding layer
    emb_mode : str, optional, default = "cell"
        The embedding mode
    device : str, optional, default = "cpu"
        The device to use. Either use "cuda" or "cpu"
    accelerator : dict, optional, default = {"cpu": True}
        The accelerator configuration

    Returns
    -------
    GeneformerConfig
        The Geneformer configuration object

    """
    def __init__(
            self, 
            batch_size: int = 5,
            emb_layer: int = -1,
            emb_mode: str = "cell",
            device: str = "cpu",
            accelerator: Optional[dict] = {"cpu": True}
            ):
        
        model_name = "geneformer-12L-30M"

        downloader = Downloader()
        downloader.download_via_name("geneformer/gene_median_dictionary.pkl")
        downloader.download_via_name("geneformer/human_gene_to_ensemble_id.pkl")
        downloader.download_via_name("geneformer/token_dictionary.pkl")
        downloader.download_via_name(f"geneformer/{model_name}/config.json")
        downloader.download_via_name(f"geneformer/{model_name}/pytorch_model.bin")
        downloader.download_via_name(f"geneformer/{model_name}/training_args.bin")
        
        self.model_dir = Path(os.path.join(downloader.CACHE_DIR_HELICAL, 'geneformer'))
        self.model_name = model_name
        self.batch_size = batch_size
        self.emb_layer = emb_layer
        self.emb_mode = emb_mode
        self.device = device
        self.accelerator = accelerator


