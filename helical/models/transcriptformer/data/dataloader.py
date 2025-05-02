import logging
import os
import random

import anndata
import numpy as np
import scanpy as sc
import torch
from scipy.sparse import csc_matrix, csr_matrix
from torch import tensor
from torch.utils.data import Dataset
from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from helical.models.transcriptformer.data.dataclasses import BatchData
from helical.models.transcriptformer.tokenizer.tokenizer import (
    BatchGeneTokenizer,
    BatchObsTokenizer,
)


def load_data(file_path):
    """Load H5AD file."""
    try:
        adata = sc.read_h5ad(file_path)
        return adata, True
    except Exception as e:
        logging.error(f"Failed to read file {file_path}: {e}")
        return None, False


def load_gene_features(adata, gene_col_name):
    """Load ensembl ids from adata object."""
    if gene_col_name == "index":
        adata.var["index"] = adata.var_names
    elif gene_col_name not in adata.var.columns:
        message = f"Gene column '{gene_col_name}' not found in adata.var.columns. Available columns: {adata.var.columns}. Modify config accordingly."
        logging.error(message)
        raise ValueError(message)
    adata = map_gene_symbols_to_ensembl_ids(adata, gene_names=gene_col_name)
    gene_names = np.array(list(adata.var["ensembl_id"].values))
    return gene_names, True


def apply_filters(
    X,
    obs,
    gene_names,
    file_path,
    filter_to_vocab,
    vocab,  # gene  vocab
    filter_outliers,
    min_expressed_genes,
):
    """Apply filters to the data."""
    if filter_to_vocab:
        filter_idx = [i for i, name in enumerate(gene_names) if name in vocab]
        X = X[:, filter_idx]
        gene_names = gene_names[filter_idx]
        if X.shape[1] == 0:
            logging.warning(f"Warning: Filtered all genes from {file_path}")
            logging.warning(f"Available genes: {len(gene_names)}")
            logging.warning(f"Number of non-zero genes: {np.sum(X > 0, axis=1).mean()}")
            return None, None, None

    if filter_outliers > 0:
        expr_counts = X.sum(axis=1)
        count_std = np.std(expr_counts)
        count_mean = np.mean(expr_counts)
        filter_idx = (expr_counts > count_mean - count_std * filter_outliers) & (
            expr_counts < count_mean + count_std * filter_outliers
        )
        X = X[filter_idx]
        obs = obs.iloc[filter_idx]

    if min_expressed_genes > 0:
        filter_idx = (X > 0).sum(axis=1) >= min_expressed_genes
        X = X[filter_idx]
        obs = obs.iloc[filter_idx]

    return X, obs, gene_names


def process_batch(
    x_batch,
    obs_batch,
    gene_names,
    gene_tokenizer,
    aux_tokenizer,
    sort_genes,
    randomize_order,
    max_len,
    pad_zeros,
    pad_token,
    gene_vocab,
    normalize_to_scale,
    clip_counts,
    aux_vocab,
):
    """Process a batch of data, including sorting, tokenization, and normalization."""
    x_batch = tensor(x_batch, dtype=torch.float32)

    # Sort genes or randomize order
    if sort_genes:
        ids_batch = torch.argsort(x_batch, dim=1, descending=True)
    else:
        ids_batch = torch.zeros_like(x_batch, dtype=torch.long)
        for i, sample in enumerate(x_batch):
            non_zero_indices = torch.nonzero(sample, as_tuple=True)[0]
            zero_indices = torch.nonzero(sample == 0, as_tuple=True)[0]
            if randomize_order:
                non_zero_indices = non_zero_indices[
                    torch.randperm(len(non_zero_indices))
                ]
                zero_indices = zero_indices[torch.randperm(len(zero_indices))]
            sample_ids = torch.cat([non_zero_indices, zero_indices])
            ids_batch[i] = sample_ids

    # Limit to max_len and gather counts
    if ids_batch.shape[1] > max_len:
        ids_batch = ids_batch[:, :max_len]

    counts_batch = torch.gather(x_batch, 1, ids_batch)

    # Tokenize gene names
    gene_names_batch = gene_names[ids_batch.numpy()]
    gene_tokens_batch = gene_tokenizer(gene_names_batch)

    # Apply padding and normalization
    if pad_zeros:
        gene_tokens_batch = gene_tokens_batch.masked_fill(
            counts_batch == 0, gene_vocab[pad_token]
        )

    # Pad ids_batch to max_len
    tok_bz, tok_sq = gene_tokens_batch.shape
    if tok_sq < max_len:
        padding = torch.full(
            (tok_bz, max_len - tok_sq),
            gene_vocab[pad_token],
            dtype=gene_tokens_batch.dtype,
        )
        gene_tokens_batch = torch.cat([gene_tokens_batch, padding], dim=1)
        gene_names_batch = np.hstack(
            [
                gene_names_batch,
                np.full((tok_bz, max_len - tok_sq), pad_token),
            ]
        )

        counts_batch = torch.cat(
            [counts_batch, torch.zeros_like(padding, dtype=counts_batch.dtype)], dim=1
        )

    # Normalize to scale if specified
    if normalize_to_scale is not None and normalize_to_scale > 0:
        row_sums = counts_batch.sum(dim=1, keepdim=True)
        counts_batch = counts_batch / row_sums * normalize_to_scale

    # Clip counts if specified
    if clip_counts is not None:
        counts_batch = counts_batch.clamp(min=0, max=clip_counts)

    # Prepare result dictionary
    result = {
        "gene_counts": counts_batch,
        "gene_token_indices": gene_tokens_batch,
    }

    # Add auxiliary and tokens if specified
    if aux_vocab is not None:
        aux_tokens_batch = torch.stack(
            [aux_tokenizer(obs) for _, obs in obs_batch.iterrows()]
        )
        result["aux_token_indices"] = aux_tokens_batch

    return result


class AnnDataset(Dataset):
    def __init__(
        self,
        files_list: list[str] | list[anndata.AnnData],
        gene_vocab: dict[str, str],
        data_dir: str = None,
        aux_vocab: dict[str, dict[str, str]] = None,
        max_len: int = 2000,
        normalize_to_scale: bool = None,
        sort_genes: bool = False,
        randomize_order: bool = False,
        pad_zeros: bool = True,
        gene_col_name: str = "feature_id",
        filter_to_vocab: bool = True,
        filter_outliers: float = 0.0,
        min_expressed_genes: int = 200,
        seed: int = 0,
        pad_token: str = "[PAD]",
        clip_counts: float = 1e10,
        inference: bool = False,
        obs_keys: list[str] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.files_list = files_list
        self.gene_vocab = gene_vocab
        self.aux_vocab = aux_vocab
        self.max_len = max_len
        self.normalize_to_scale = normalize_to_scale
        self.sort_genes = sort_genes
        self.randomize_order = randomize_order
        self.pad_zeros = pad_zeros
        self.gene_col_name = gene_col_name
        self.filter_to_vocab = filter_to_vocab
        self.filter_outliers = filter_outliers
        self.min_expressed_genes = min_expressed_genes
        self.seed = seed
        self.pad_token = pad_token
        self.clip_counts = clip_counts
        self.inference = inference
        self.obs_keys = obs_keys

        self.gene_tokenizer = BatchGeneTokenizer(gene_vocab)
        if aux_vocab is not None:
            self.aux_tokenizer = BatchObsTokenizer(aux_vocab)

        random.seed(self.seed)

        logging.info("Loading and processing all data")
        self.data = self.load_and_process_all_data()

    def _get_batch_from_file(self, file: str | anndata.AnnData) -> BatchData | None:
        if isinstance(file, str):
            file_path = (
                file if self.data_dir is None else os.path.join(self.data_dir, file)
            )
            adata, success = load_data(file_path)
        elif isinstance(file, anndata.AnnData):
            adata = file
            success = True
            file_path = None
        else:
            raise ValueError(f"Invalid file type: {type(file)}")

        if not success:
            logging.error(
                f"Failed to load data from {file_path if file_path else 'provided AnnData object'}"
            )
            return None

        gene_names, success = load_gene_features(adata, self.gene_col_name)
        if not success:
            logging.error(
                f"Failed to load gene features from {file_path if file_path else 'provided AnnData object'}"
            )
            return None

        X = (
            adata.X.toarray()
            if isinstance(adata.X, csr_matrix | csc_matrix)
            else adata.X
        )
        obs = adata.obs

        if not hasattr(obs, "assay"):
            logging.warning(
                f"'assay' column not found in {file_path if file_path else 'provided AnnData object'}. Adding 'unknown' as default."
            )
            obs["assay"] = "unknown"

        vocab = self.gene_vocab
        X, obs, gene_names = apply_filters(
            X,
            obs,
            gene_names,
            file_path,
            self.filter_to_vocab,
            vocab,
            self.filter_outliers,
            self.min_expressed_genes,
        )
        if X is None:
            logging.warning(
                f"Data was filtered out completely for {file_path if file_path else 'provided AnnData object'}"
            )
            return None

        batch = process_batch(
            X,
            obs,
            gene_names,
            self.gene_tokenizer,
            getattr(self, "aux_tokenizer", None),
            self.sort_genes,
            self.randomize_order,
            self.max_len,
            self.pad_zeros,
            self.pad_token,
            self.gene_vocab,
            self.normalize_to_scale,
            self.clip_counts,
            self.aux_vocab,
        )
        batch["file_path"] = (
            np.array([file_path] * X.shape[0]) if file_path is not None else None
        )

        if self.obs_keys is not None:
            obs_data = {}
            if "all" in self.obs_keys:
                # Keep all columns from obs
                self.obs_keys = obs.columns
                for col in obs.columns:
                    obs_data[col] = np.array(obs[col].tolist())[:, None]
            else:
                # Keep only specified columns
                for col in self.obs_keys:
                    obs_data[col] = np.array(obs[col].tolist())[:, None]
            batch["obs"] = obs_data

        return BatchData(**batch)

    def load_and_process_all_data(self):
        all_data = []
        for i, file in enumerate(self.files_list):
            logging.info(f"Processing validation file {i+1} of {len(self.files_list)}")
            file_batch = self._get_batch_from_file(file)
            if file_batch is None:
                continue

            all_data.append(file_batch)

        # Add check for empty all_data list
        if not all_data:
            raise ValueError(
                "No valid data was loaded from any files. "
                "Check if files exist and contain valid data after filtering."
            )

        concatenated_batch = BatchData(
            gene_counts=torch.concat([batch.gene_counts for batch in all_data]),
            gene_token_indices=torch.concat(
                [batch.gene_token_indices for batch in all_data]
            ),
            file_path=None,
            aux_token_indices=(
                torch.concat([batch.aux_token_indices for batch in all_data])
                if all_data[0].aux_token_indices is not None
                else None
            ),
            obs=(
                {
                    col: np.vstack([batch.obs[col] for batch in all_data])
                    for col in self.obs_keys
                }
                if self.obs_keys is not None
                else None
            ),
        )

        return concatenated_batch

    def __len__(self):
        return len(self.data.gene_counts)

    def __getitem__(self, idx):
        data_dict = {}
        for key, value in self.data.__dict__.items():
            if value is None:
                data_dict[key] = None
            elif isinstance(value, dict):
                data_dict[key] = {k: v[idx] for k, v in value.items()}
            else:
                data_dict[key] = value[idx]
        return BatchData(**data_dict)

    @staticmethod
    def collate_fn(batch: BatchData | list[BatchData]) -> BatchData:
        if isinstance(batch, BatchData):
            return batch

        collated_batch = BatchData(
            gene_counts=torch.stack([item.gene_counts for item in batch]),
            gene_token_indices=torch.stack([item.gene_token_indices for item in batch]),
            file_path=None,
            aux_token_indices=(
                torch.stack([item.aux_token_indices for item in batch])
                if batch[0].aux_token_indices is not None
                else None
            ),
            obs=(
                {
                    col: np.vstack([item.obs[col] for item in batch])
                    for col in batch[0].obs.keys()
                }
                if batch[0].obs is not None
                else None
            ),
        )
        return collated_batch
