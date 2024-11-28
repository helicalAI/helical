from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL
from typing import Literal
import logging

LOGGER = logging.getLogger(__name__)

class CaduceusConfig():
    def __init__(
        self, 
        model_name: Literal["caduceus-ph-4L-seqlen-1k-d118", "caduceus-ph-4L-seqlen-1k-d256", "caduceus-ph-16L-seqlen-131k-d256", "caduceus-ps-4L-seqlen-1k-d118", "caduceus-ps-4L-seqlen-1k-d256", "caduceus-ps-16L-seqlen-131k-d256"] = "caduceus-ph-4L-seqlen-1k-d118",
        batch_size: int = 5,
        device: Literal["cpu", "cuda"] = "cpu",
        nproc: int = 1,
        ):

        model_map = {
            "caduceus-ph-4L-seqlen-1k-d118": {
                "input_size": 1024,
                "embedding_size": 118,
            },
            "caduceus-ph-4L-seqlen-1k-d256": {
                "input_size": 1024,
                "embedding_size": 256,
            },
            "caduceus-ph-16L-seqlen-131k-d256": {
                "input_size": 1024*128,
                "embedding_size": 256,
            },
            "caduceus-ps-4L-seqlen-1k-d118": {
                "input_size": 1024,
                "embedding_size": 118,
            },
            "caduceus-ps-4L-seqlen-1k-d256": {
                "input_size": 1024,
                "embedding_size": 256,
            },
            "caduceus-ps-16L-seqlen-131k-d256": {
                "input_size": 1024*128,
                "embedding_size": 256,
            }
        }

        if model_name not in model_map.keys():
            error = f"Model name {model_name} not found in available models: {model_map.keys()}"
            LOGGER.error(error)
            raise ValueError(error)

        self.list_of_files_to_download = [
            f"caduceus/{model_name}/model.safetensors",
            f"caduceus/{model_name}/config.json"
        ]

        self.files_config = {
            "model_files_dir": Path(CACHE_DIR_HELICAL, 'caduceus', model_name),
        }

        self.config = {
            "input_size": model_map[model_name]["input_size"],
            "embedding_size": model_map[model_name]["embedding_size"],
            "model_name": model_name,
            "batch_size": batch_size,   
            "device": device,
            "nproc": nproc,
        }
