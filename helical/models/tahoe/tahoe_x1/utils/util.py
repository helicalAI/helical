# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import logging
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from scanpy import AnnData
from scipy.sparse import csc_matrix, csr_matrix

from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab

log = logging.getLogger(__name__)


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


# Wrapper function for S3 downloads used by other modules
from helical.models.tahoe.tahoe_x1.utils.s3_utils import download_file_from_s3


def download_file_from_s3_url(s3_url, local_file_path):
    """Downloads a file from an S3 URL to the specified local path.

    Supports public S3 buckets without credentials (like --no-sign-request).

    :param s3_url: S3 URL in the form s3://bucket-name/path/to/file
    :param local_file_path: Local path where the file will be saved.
    :return: The local path to the downloaded file, or None if download fails.
    """
    return download_file_from_s3(s3_url, local_file_path)
