import argparse as ap
import glob
import logging
import os
from helical.models.state._embed_utils.inference import Inference
import torch
from omegaconf import OmegaConf
import numpy as np
from helical.models.base_models import HelicalBaseFoundationModel
from helical.models.state.state_config import stateConfig
from helical.utils.downloader import Downloader
from huggingface_hub import hf_hub_download, snapshot_download

LOGGER = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# this code to do embedding generation
class stateEmbeddingsModel(HelicalBaseFoundationModel):
    def __init__(self, configurer: stateConfig = None) -> None:
        super().__init__()

        if configurer is None:
            configurer = stateConfig()

        self.config = configurer.config["embed"]

        # downloader = Downloader()
        # # we need to download the model weights for SE
        # downloader.download_via_name(self.config["list_of_files_to_download"])

        # Download to helical cache directory
        # local_dir = snapshot_download(
        #     repo_id=self.config["repo_id"],
        #     local_dir=self.config["model_dir"],      # where to put everything
        #     local_dir_use_symlinks=False             # make real copies instead of symlinks
        # )
        local_dir = snapshot_download(
            repo_id=self.config["repo_id"],
            local_dir=self.config["model_dir"],
            local_dir_use_symlinks=False,
            allow_patterns=[
                self.config["filename"],
                "config.yaml",
                "protein_embeddings.pt",
            ]
        )
        ckpt_path = os.path.join(self.config["model_dir"], self.config["filename"])

        LOGGER.info(f"Using model checkpoint: {ckpt_path}")

        # Create inference object
        embedding_file = os.path.join(
            self.config["model_dir"], "protein_embeddings.pt"
        )
        protein_embeds = torch.load(
            embedding_file, weights_only=False, map_location="cpu"
        )

        self.model_conf = OmegaConf.load(
            os.path.join(self.config["model_dir"], "config.yaml")
        )

        self.embed_model = Inference(cfg=self.model_conf, protein_embeds=protein_embeds)
        self.embed_model.load_model(ckpt_path)

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(self.config["output_path"])
        os.makedirs(output_dir, exist_ok=True)
        LOGGER.info(f"Created output directory: {self.config['output_path']}")

    def process_data(self, ann_data_path):
        dataloader = self.embed_model.process_data(
            input_adata_path=ann_data_path,
        )
        LOGGER.info("Successfully processed the data for State Embeddings.")
        return dataloader

    def get_embeddings(self, dataloader) -> np.ndarray:
        embeddings = self.embed_model.embed_data(
            dataloader=dataloader,
        )
        LOGGER.info("Finished getting embeddings.")
        return np.concatenate(embeddings)
