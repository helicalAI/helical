import logging
import numpy as np
from anndata import AnnData
from torch.utils.data import DataLoader
import os
from pathlib import Path
from helical.models.hyena_dna.hyena_dna_config import HyenaDNAConfig
from helical.models.helical import HelicalBaseModel
from helical.models.uce.uce_utils import get_ESM2_embeddings, load_model, process_data, get_gene_embeddings
from accelerate import Accelerator
from helical.services.downloader import Downloader
from typing import Optional

class HyenaDNA(HelicalBaseModel):
    """HyenaDNA model."""
    default_configurer = HyenaDNAConfig()

    def __init__(self, model_dir: Optional[str] = None, configurer: HyenaDNAConfig = default_configurer) -> None:    
        super().__init__()
        self.config = configurer.config
        self.log = logging.getLogger("Hyena-DNA-Model")
        
        if model_dir is None: 
            self.downloader = Downloader()
            model_path = f"hyena_dna/{self.config['model_name']}.ckpt"
            self.downloader.download_via_name(model_path)
            self.model_path = Path(os.path.join(self.downloader.CACHE_DIR_HELICAL, model_path))
        else:
            self.model_path = Path(os.path.join(model_dir, f"{self.config['model_name']}.ckpt"))
        
        

    def process_data(self):
        pass

    def get_embeddings(self):
        pass