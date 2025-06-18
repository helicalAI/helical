import logging
import anndata
import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from helical.models.transcriptformer.data.dataloader import AnnDataset
from helical.models.transcriptformer.model_dir.embedding_surgery import (
    change_embedding_layer,
)
from helical.models.transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
from helical.models.transcriptformer.utils.utils import stack_dict
from helical.models.base_models import HelicalRNAModel
from helical.utils.downloader import Downloader
from omegaconf import OmegaConf
import json
import os
import pandas as pd
from helical.constants.paths import CACHE_DIR_HELICAL
from helical.models.transcriptformer.transcriptformer_config import (
    TranscriptFormerConfig,
)
from tqdm import tqdm

torch.set_float32_matmul_precision("high")

logger = logging.getLogger(__name__)


class TranscriptFormer(HelicalRNAModel):
    """
    TranscriptFormer model for RNA-seq data.
    This class is a wrapper around the TranscriptFormer model and provides methods for
    processing data, getting embeddings, and loading the model.

    Parameters
    ----------
    configurer: TranscriptFormerConfig
        Configuration object for the TranscriptFormer model.

    Example
    -------
    ```python
    from helical.models.transcriptformer.model import TranscriptFormer
    from helical.models.transcriptformer.transcriptformer_config import TranscriptFormerConfig
    import anndata as ad

    configurer = TranscriptFormerConfig()
    model = TranscriptFormer(configurer)

    ann_data = ad.read_h5ad("/path/to/data.h5ad")

    dataset = model.process_data([ann_data])
    embeddings = model.get_embeddings(dataset)
    print(embeddings)
    ```

    Notes
    -----
    We use the implementation from [this repository](https://github.com/czi-ai/transcriptformer),
    which comes from the original authors. You can find the description of the method in
    [this paper](https://www.biorxiv.org/content/10.1101/2025.04.25.650731v1).
    """

    configurer = TranscriptFormerConfig()

    def __init__(self, configurer: TranscriptFormerConfig = configurer):
        super().__init__(configurer.config)
        self.config = configurer.config

        downloader = Downloader()
        for file in configurer.list_of_files_to_download:
            downloader.download_via_name(file)

        logger.info(f"Loading cache config for {configurer.model_name}")
        cache_config_path = os.path.join(
            CACHE_DIR_HELICAL, "transcriptformer", configurer.model_name, "config.json"
        )
        with open(cache_config_path) as f:
            cache_config_dict = json.load(f)
        cache_config = OmegaConf.create(cache_config_dict)

        # Merge the cache config with the config provided by the user
        self.config = OmegaConf.merge(cache_config, self.config)

        self.config.model.inference_config.load_checkpoint = os.path.join(
            CACHE_DIR_HELICAL,
            "transcriptformer",
            configurer.model_name,
            "model_weights.pt",
        )
        self.config.model.data_config.aux_vocab_path = os.path.join(
            CACHE_DIR_HELICAL, "transcriptformer", configurer.model_name, "vocabs"
        )
        self.config.model.data_config.aux_cols = "assay"
        self.config.model.data_config.esm2_mappings_path = os.path.join(
            CACHE_DIR_HELICAL, "transcriptformer", configurer.model_name, "vocabs"
        )
        logger.info(f"Merged cache config with user config for {configurer.model_name}")

        logger.debug(OmegaConf.to_yaml(self.config))

        # Load vocabs and embeddings
        (self.gene_vocab, self.aux_vocab), self.emb_matrix = load_vocabs_and_embeddings(
            self.config
        )

        # Instantiate the model
        logger.info("Instantiating the model")
        self.model = instantiate(
            self.config.model,
            gene_vocab_dict=self.gene_vocab,
            aux_vocab_dict=self.aux_vocab,
            emb_matrix=self.emb_matrix,
            emb_mode=self.config.model.inference_config.emb_mode,
        )
        self.model.eval()

        logger.info("Model instantiated successfully")

        # Check if checkpoint is supplied
        if (
            not hasattr(self.model.inference_config, "load_checkpoint")
            or not self.model.inference_config.load_checkpoint
        ):
            raise ValueError(
                "No checkpoint provided for inference. Please specify a checkpoint path in "
                "model.inference_config.load_checkpoint"
            )

        logger.info("Loading model checkpoint")
        state_dict = torch.load(
            self.model.inference_config.load_checkpoint, weights_only=True
        )

        # Filter out auxiliary embedding weights if aux_vocab_path is None
        if self.model.data_config.aux_vocab_path is None:
            filtered_state_dict = {
                k: v
                for k, v in state_dict.items()
                if not k.startswith("aux_embeddings.")
            }
            state_dict = filtered_state_dict

        self.model.load_state_dict(state_dict)
        logger.info("Model weights loaded successfully")

        # Perform embedding surgery if specified in config
        if self.model.inference_config.pretrained_embedding is not None:
            logger.info("Performing embedding surgery")
            # Check if pretrained_embedding_paths is a list, if not convert it to a list
            if not isinstance(self.model.inference_config.pretrained_embedding, list):
                pretrained_embedding_paths = [
                    self.model.inference_config.pretrained_embedding
                ]
            else:
                pretrained_embedding_paths = (
                    self.model.inference_config.pretrained_embedding
                )
            self.model, self.gene_vocab = change_embedding_layer(
                self.model, pretrained_embedding_paths
            )

        mode = "training" if self.model.training else "eval"
        logger.info(
            f"TranscriptFormer '{configurer.model_name}' model is in '{mode}' mode, "
            f"on device GPU generating embeddings for '{self.model.inference_config.output_keys}' & {self.model.emb_mode} embeddings."
        )

    def process_data(self, data_files: list[str] | list[anndata.AnnData]):
        """
        Process the data for TranscriptFormer.

        Parameters
        ----------
        data_files: list[str] | list[anndata.AnnData]
            List of paths to AnnData files or AnnData objects.

        Returns
        -------
            dataset: The processed dataset.
        """
        # Load dataset

        logger.info(f"Processing data for TranscriptFormer.")

        data_kwargs = {
            "gene_vocab": self.gene_vocab,
            "aux_vocab": self.aux_vocab,
            "max_len": self.model.model_config.seq_len,
            "pad_zeros": self.model.data_config.pad_zeros,
            "pad_token": self.model.data_config.gene_pad_token,
            "sort_genes": self.model.data_config.sort_genes,
            "filter_to_vocab": self.model.data_config.filter_to_vocabs,
            "filter_outliers": self.model.data_config.filter_outliers,
            "gene_col_name": self.model.data_config.gene_col_name,
            "normalize_to_scale": self.model.data_config.normalize_to_scale,
            "randomize_order": self.model.data_config.randomize_genes,
            "min_expressed_genes": self.model.data_config.min_expressed_genes,
            "clip_counts": self.model.data_config.clip_counts,
            "obs_keys": self.model.inference_config.obs_keys,
        }
        dataset = AnnDataset(data_files, **data_kwargs)
        return dataset

    def get_embeddings(self, dataset: AnnDataset) -> torch.Tensor:
        """
        Get the embeddings for the dataset.

        Parameters
        ----------
        dataset: AnnDataset
            The dataset to get the embeddings for.

        Returns
        -------
        embeddings: torch.Tensor
            The embeddings for the dataset.
        """
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.model.inference_config.batch_size,
            num_workers=self.model.data_config.n_data_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        output = []
        progress_bar = tqdm(
            dataloader,
            desc="Embedding Cells",
            total=len(dataloader),
            unit="batch",
        )

        self.model.to(self.config.model.inference_config.device)
        with torch.no_grad():
            for batch in progress_bar:
                output_batch = self.model.inference(batch)
                output.append(output_batch)

        logger.info("Combining predictions")
        concat_output = stack_dict(output)

        # Create pandas DataFrames from the obs and uns data in concat_output
        obs_df = pd.DataFrame(concat_output["obs"])
        uns = (
            {"llh": pd.DataFrame({"llh": concat_output["llh"]})}
            if "llh" in concat_output
            else None
        )
        obsm = {}

        # Add all other output keys to the obsm
        for k in self.model.inference_config.output_keys:
            if k in concat_output:
                obsm[k] = concat_output[k].numpy()

        # Create a new AnnData object with the embeddings
        self.output_adata = anndata.AnnData(
            obs=obs_df,
            obsm=obsm,
            uns=uns,
        )

        logger.info(f"Returning '{self.model.emb_mode}_embeddings' from output.")
        embeddings = concat_output[self.model.emb_mode + "_embeddings"]
        return embeddings

    def get_output_adata(self) -> anndata.AnnData:
        """
        Get the output AnnData object. Only call this after running 'get_embeddings'.

        Returns
        -------
            output_adata: The output AnnData object.
        """
        if not hasattr(self, "output_adata"):
            message = (
                "Output AnnData object not found. Please run 'get_embeddings' first."
            )
            logger.error(message)
            raise ValueError(message)
        logger.info(
            "Returning output AnnData object, embeddings are stored in .obsm['embeddings'], uns['llh'] contains the log-likelihoods"
        )
        return self.output_adata
