from typing import Optional
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal

class HelixRConfig:
    def __init__(self, 
            model_name: Literal["helixR-16L"] = "helixR-16L",
            batch_size: int = 24,
            device: Literal["cpu", "cuda"] = "cpu",
            nproc: int = 1):
        
        self.model_dir = Path(CACHE_DIR_HELICAL, 'helixr')

        self.list_of_files_to_download = [
            f"geneformer/{model_name}/config.json ",
            f"geneformer/{model_name}/training_args.bin",
            f"geneformer/{model_name}/model.safetensors",
            f"geneformer/{model_name}/generation_config.json",
        ]

        self.config = {
            'model_name': model_name,
            'batch_size': batch_size,
            'device': device,
            'nproc': nproc
        }
