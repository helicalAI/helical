import argparse as ap
import glob
import logging
import os
from .model_dir._embed_utils.inference import Inference
import torch
from omegaconf import OmegaConf
import numpy as np
from helical.models.base_models import HelicalBaseFoundationModel
from helical.models.state.state_config import stateConfig
from helical.utils.downloader import Downloader


LOGGER = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# this code to do embedding generation
class stateEmbeddingsModel(HelicalBaseFoundationModel):
    def __init__(self, configurer: stateConfig = None) -> None:
        super().__init__()

        if configurer is None:
            configurer = stateConfig()

        self.config = configurer.config["embed"]

        Downloader().download_via_name(self.config["list_of_files_to_download"])

        self.model_dir = self.config["cache_dir"]
        ckpt_path = os.path.join(self.model_dir, self.config["embed_checkpoint"])

        LOGGER.info(f"Using model checkpoint: {ckpt_path}")

        # Create inference object
        embedding_file = os.path.join(
            self.model_dir, "protein_embeddings.pt"
        )
        protein_embeds = torch.load(
            embedding_file, weights_only=False, map_location="cpu"
        )

        self.model_conf = OmegaConf.load(
            os.path.join(self.model_dir, "config.yaml")
        )

        self.embed_model = Inference(cfg=self.model_conf, protein_embeds=protein_embeds)
        self.embed_model.load_model(ckpt_path)

    def process_data(self, adata):
        dataloader = self.embed_model.process_data(
            adata=adata,
        )
        LOGGER.info("Successfully processed the data for State Embeddings.")
        return dataloader

    def get_embeddings(self, dataloader) -> np.ndarray:
        embeddings = self.embed_model.embed_data(
            dataloader=dataloader,
        )
        LOGGER.info("Finished getting embeddings.")
        return np.concatenate(embeddings)
