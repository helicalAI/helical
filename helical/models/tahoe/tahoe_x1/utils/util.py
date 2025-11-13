# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
from pathlib import Path

log = logging.getLogger(__name__)
import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from scanpy import AnnData
from scipy.sparse import csc_matrix, csr_matrix
from scipy.stats import pearsonr
from sklearn.neighbors import kneighbors_graph

from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab


def load_model(
    model_dir: str,
    device: torch.device,
    return_gene_embeddings: bool = False,
    use_chem_inf: Optional[bool] = False,
):
    from helical.models.tahoe.tahoe_x1.model.model import ComposerTX

    model_config_path = os.path.join(model_dir, "model_config.yml")
    vocab_path = os.path.join(model_dir, "vocab.json")
    collator_config_path = os.path.join(model_dir, "collator_config.yml")
    ckpt = os.path.join(model_dir, "best-model.pt")

    model_config = om.load(model_config_path)
    if model_config["attn_config"]["attn_impl"] == "triton":
        model_config["attn_config"]["attn_impl"] = "flash"
        model_config["attn_config"]["use_attn_mask"] = False

    model_config["do_mlm"] = False  # Disable MLM for embeddings generation
    model_config["return_gene_embeddings"] = return_gene_embeddings

    collator_config = om.load(collator_config_path)
    vocab = GeneVocab.from_file(vocab_path)

    # handle chemical information
    strict = True

    # if model was trained with chemical information, and we don't want to use it for inference
    if use_chem_inf is not None and (
        not use_chem_inf and collator_config.get("use_chem_token", False)
    ):
        # we need to modify the model and collator config accordingly
        collator_config["use_chem_token"] = False
        del model_config["chemical_encoder"]
        del collator_config["drug_to_id_path"]
        strict = False

    model = ComposerTX(
        model_config=model_config,
        collator_config=collator_config,
    )
    model.load_state_dict(
        torch.load(ckpt, weights_only=False)["state"]["model"],
        strict=strict,
    )
    model.to(device)
    model.eval()
    log.info(f"Model loaded from {ckpt}")

    return model, vocab, model_config, collator_config


def loader_from_adata(
    adata: AnnData,
    collator_cfg: DictConfig,
    vocab: GeneVocab,
    batch_size: int = 50,
    max_length: Optional[int] = None,
    gene_ids: Optional[np.ndarray] = None,
    num_workers: int = 8,
    prefetch_factor: int = 48,
):
    count_matrix = adata.X
    if isinstance(count_matrix, np.ndarray):
        count_matrix = csr_matrix(count_matrix)
    elif isinstance(count_matrix, csc_matrix):
        count_matrix = count_matrix.tocsr()
    elif hasattr(count_matrix, "to_memory"):
        count_matrix = count_matrix.to_memory().tocsr()

    if gene_ids is None:
        gene_ids = np.array(adata.var["id_in_vocab"])
        assert np.all(gene_ids >= 0)

    if max_length is None:
        max_length = len(gene_ids)

    from helical.models.tahoe.tahoe_x1.data import CountDataset, DataCollator

    dataset = CountDataset(
        count_matrix,
        gene_ids,
        cls_token_id=vocab["<cls>"],
        pad_value=collator_cfg["pad_value"],
    )
    collate_fn = DataCollator(
        vocab=vocab,
        drug_to_id_path=collator_cfg.get("drug_to_id_path", None),
        do_padding=collator_cfg.get("do_padding", True),
        unexp_padding=False,  # Disable padding with random unexpressed genes for inference
        pad_token_id=collator_cfg.pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=False,  # Disable masking for inference
        do_binning=collator_cfg.get("do_binning", True),
        log_transform=collator_cfg.get("log_transform", False),
        target_sum=collator_cfg.get("target_sum"),
        mlm_probability=collator_cfg.mlm_probability,  # Not used
        mask_value=collator_cfg.mask_value,
        max_length=max_length,
        sampling=collator_cfg.sampling,  # Turned on since max-length can be less than the number of genes
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
        keep_first_n_tokens=collator_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=collator_cfg.get("use_chem_token", False),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )

    return data_loader


def compute_lisi_scores(emb, labels, k):
    """Compute LISI (Local Inverse Simpson's Index) scores for embeddings.

    Args:
        emb: Embedding matrix of shape (n_cells, n_features)
        labels: Cell type labels for each cell
        k: Number of neighbors to consider

    Returns:
        LISI score normalized by theoretical maximum
    """
    nng = kneighbors_graph(emb, n_neighbors=k).tocoo()
    labels = pd.Categorical(labels).codes
    self_id = labels[nng.row]
    ne_id = labels[nng.col]

    _, c = np.unique(labels, return_counts=True)
    theoretic_score = ((c / c.sum()) ** 2).sum()
    return (self_id == ne_id).mean() / theoretic_score


def add_file_handler(logger: logging.Logger, log_file_path: Path):
    """Add a file handler to the logger."""
    h = logging.FileHandler(log_file_path)

    # format showing time, name, function, and message
    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s-%(funcName)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    h.setFormatter(formatter)
    h.setLevel(logger.level)
    logger.addHandler(h)


from helical.models.tahoe.tahoe_x1.utils.s3_utils import download_file_from_s3


def download_file_from_s3_url(s3_url, local_file_path):
    """Downloads a file from an S3 URL to the specified local path.

    Supports public S3 buckets without credentials (like --no-sign-request).

    :param s3_url: S3 URL in the form
    s3://bucket-name/path/to/file
    :param local_file_path: Local path where the file will be saved.
    :return: The local path to the downloaded file, or None if download fails.
    """
    # Delegate to the new S3 utilities module
    return download_file_from_s3(s3_url, local_file_path)


def calc_pearson_metrics(preds, targets, conditions, mean_ctrl):

    conditions_unique = np.unique(conditions)
    condition2idx = {c: np.where(conditions == c)[0] for c in conditions_unique}

    targets_mean_perturbed_by_condition = np.array(
        [targets[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    preds_mean_perturbed_by_condition = np.array(
        [preds[condition2idx[c]].mean(0) for c in conditions_unique],
    )  # (n_conditions, n_genes)

    pearson = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        print(cond, pearsonr(t, p))
        pearson.append(pearsonr(t, p)[0])

    pearson_delta = []
    for cond, t, p in zip(
        conditions_unique,
        targets_mean_perturbed_by_condition,
        preds_mean_perturbed_by_condition,
    ):
        tm, pm = t, p
        tm -= mean_ctrl
        pm -= mean_ctrl

        print(cond, pearsonr(tm, pm))
        pearson_delta.append(pearsonr(tm, pm)[0])

    return {
        "pearson": np.mean(pearson),
        "pearson_delta": np.mean(pearson_delta),
    }
