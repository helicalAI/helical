import logging

import anndata
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from helical.models.transcriptformer.data.dataloader import AnnDataset
from helical.models.transcriptformer.model_dir.embedding_surgery import change_embedding_layer
from helical.models.transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
from helical.models.transcriptformer.utils.utils import stack_dict
from helical.models.base_models import HelicalRNAModel
from omegaconf import DictConfig
# Set float32 matmul precision for better performance with Tensor Cores
torch.set_float32_matmul_precision("high")

torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.cache_size_limit = 1000

class TranscriptFormer(HelicalRNAModel):

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        # Load vocabs and embeddings
        (self.gene_vocab, self.aux_vocab), self.emb_matrix = load_vocabs_and_embeddings(cfg)

        # Instantiate the model
        logging.info("Instantiating the model")
        self.model = instantiate(
            cfg.model,
            gene_vocab_dict=self.gene_vocab,
            aux_vocab_dict=self.aux_vocab,
            emb_matrix=self.emb_matrix,
        )
        self.model.eval()

        logging.info("Model instantiated successfully")

        # Check if checkpoint is supplied
        if not hasattr(self.model.inference_config, "load_checkpoint") or not self.model.inference_config.load_checkpoint:
            raise ValueError(
                "No checkpoint provided for inference. Please specify a checkpoint path in "
                "model.inference_config.load_checkpoint"
            )

        logging.info("Loading model checkpoint")
        # Instead of loading full checkpoint, just load weights
        state_dict = torch.load(self.model.inference_config.load_checkpoint, weights_only=True)

        # Validate and load weights
        # converter.validate_loaded_weights(model, state_dict)
        
        # Filter out auxiliary embedding weights if aux_vocab_path is None
        if self.model.data_config.aux_vocab_path is None:
            filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_embeddings.')}
            state_dict = filtered_state_dict
        
        self.model.load_state_dict(state_dict)
        logging.info("Model weights loaded successfully")

        # Perform embedding surgery if specified in config
        if self.model.inference_config.pretrained_embedding is not None:
            logging.info("Performing embedding surgery")
            # Check if pretrained_embedding_paths is a list, if not convert it to a list
            if not isinstance(self.model.inference_config.pretrained_embedding, list):
                pretrained_embedding_paths = [self.model.inference_config.pretrained_embedding]
            else:
                pretrained_embedding_paths = self.model.inference_config.pretrained_embedding
            self.model, self.gene_vocab = change_embedding_layer(self.model, pretrained_embedding_paths)

    def process_data(self, data_files: list[str] | list[anndata.AnnData]):
        # Load dataset
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
    
    def get_embeddings(self, dataset: AnnDataset):
        # Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.model.inference_config.batch_size,
            num_workers=self.model.data_config.n_data_workers,
            drop_last=False,
            pin_memory=True,
            collate_fn=dataset.collate_fn,
        )

        # Create Trainer
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,  # Multiple GPUs/nodes not supported for inference
            num_nodes=1,
            precision=self.model.inference_config.precision,
            limit_predict_batches=None,
            logger=CSVLogger("logs", name="inference"),
        )

        # Run prediction
        output = trainer.predict(self.model, dataloaders=dataloader)

        # Combine predictions
        logging.info("Combining predictions")
        concat_output = stack_dict(output)

        # Create pandas DataFrames from the obs and uns data in concat_output
        obs_df = pd.DataFrame(concat_output["obs"])
        uns = {"llh": pd.DataFrame({"llh": concat_output["llh"]})} if "llh" in concat_output else None
        obsm = {}

        # Add all other output keys to the obsm
        for k in self.model.inference_config.output_keys:
            if k in concat_output:
                obsm[k] = concat_output[k].numpy()

        # Create a new AnnData object with the embeddings
        output_adata = anndata.AnnData(
            obs=obs_df,
            obsm=obsm,
            uns=uns,
        )

        return output_adata
