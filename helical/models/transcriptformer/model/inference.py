import logging

import anndata
import pandas as pd
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader

from helical.models.transcriptformer.data.dataloader import AnnDataset
from helical.models.transcriptformer.model.embedding_surgery import change_embedding_layer
from helical.models.transcriptformer.tokenizer.vocab import load_vocabs_and_embeddings
from helical.models.transcriptformer.utils.utils import stack_dict

# Set float32 matmul precision for better performance with Tensor Cores
torch.set_float32_matmul_precision("high")

torch._dynamo.config.optimize_ddp = False
torch._dynamo.config.cache_size_limit = 1000


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def run_inference(cfg, data_files: list[str] | list[anndata.AnnData]):
    """Run inference using the provided config and AnnData object.

    Args:
        cfg: OmegaConf configuration object
        data_files: list of data files to load
    Returns:
        AnnData: Processed data with embeddings and likelihood scores
    """
    # Load vocabs and embeddings
    (gene_vocab, aux_vocab), emb_matrix = load_vocabs_and_embeddings(cfg)

    # Instantiate the model
    logging.info("Instantiating the model")
    model = instantiate(
        cfg.model,
        gene_vocab_dict=gene_vocab,
        aux_vocab_dict=aux_vocab,
        emb_matrix=emb_matrix,
    )
    model.eval()

    logging.info("Model instantiated successfully")

    # Check if checkpoint is supplied
    if not hasattr(cfg.model.inference_config, "load_checkpoint") or not cfg.model.inference_config.load_checkpoint:
        raise ValueError(
            "No checkpoint provided for inference. Please specify a checkpoint path in "
            "model.inference_config.load_checkpoint"
        )

    logging.info("Loading model checkpoint")
    # Instead of loading full checkpoint, just load weights
    state_dict = torch.load(cfg.model.inference_config.load_checkpoint, weights_only=True)

    # Validate and load weights
    # converter.validate_loaded_weights(model, state_dict)
    
    # Filter out auxiliary embedding weights if aux_vocab_path is None
    if cfg.model.data_config.aux_vocab_path is None:
        filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_embeddings.')}
        state_dict = filtered_state_dict
    
    model.load_state_dict(state_dict)
    logging.info("Model weights loaded successfully")

    # Perform embedding surgery if specified in config
    if cfg.model.inference_config.pretrained_embedding is not None:
        logging.info("Performing embedding surgery")
        # Check if pretrained_embedding_paths is a list, if not convert it to a list
        if not isinstance(cfg.model.inference_config.pretrained_embedding, list):
            pretrained_embedding_paths = [cfg.model.inference_config.pretrained_embedding]
        else:
            pretrained_embedding_paths = cfg.model.inference_config.pretrained_embedding
        model, gene_vocab = change_embedding_layer(model, pretrained_embedding_paths)

    # Load dataset
    data_kwargs = {
        "gene_vocab": gene_vocab,
        "aux_vocab": aux_vocab,
        "max_len": cfg.model.model_config.seq_len,
        "pad_zeros": cfg.model.data_config.pad_zeros,
        "pad_token": cfg.model.data_config.gene_pad_token,
        "sort_genes": cfg.model.data_config.sort_genes,
        "filter_to_vocab": cfg.model.data_config.filter_to_vocabs,
        "filter_outliers": cfg.model.data_config.filter_outliers,
        "gene_col_name": cfg.model.data_config.gene_col_name,
        "normalize_to_scale": cfg.model.data_config.normalize_to_scale,
        "randomize_order": cfg.model.data_config.randomize_genes,
        "min_expressed_genes": cfg.model.data_config.min_expressed_genes,
        "clip_counts": cfg.model.data_config.clip_counts,
        "obs_keys": cfg.model.inference_config.obs_keys,
    }
    dataset = AnnDataset(data_files, **data_kwargs)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.model.inference_config.batch_size,
        num_workers=cfg.model.data_config.n_data_workers,
        drop_last=False,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    # Create Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,  # Multiple GPUs/nodes not supported for inference
        num_nodes=1,
        precision=cfg.model.inference_config.precision,
        limit_predict_batches=None,
        logger=CSVLogger("logs", name="inference"),
    )

    # Run prediction
    output = trainer.predict(model, dataloaders=dataloader)

    # Combine predictions
    logging.info("Combining predictions")
    concat_output = stack_dict(output)

    # Create pandas DataFrames from the obs and uns data in concat_output
    obs_df = pd.DataFrame(concat_output["obs"])
    uns = {"llh": pd.DataFrame({"llh": concat_output["llh"]})} if "llh" in concat_output else None
    obsm = {}

    # Add all other output keys to the obsm
    for k in cfg.model.inference_config.output_keys:
        if k in concat_output:
            obsm[k] = concat_output[k].numpy()

    # Create a new AnnData object with the embeddings
    output_adata = anndata.AnnData(
        obs=obs_df,
        obsm=obsm,
        uns=uns,
    )

    return output_adata
