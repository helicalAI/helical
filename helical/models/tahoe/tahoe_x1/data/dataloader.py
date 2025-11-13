# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
from collections.abc import MutableSequence
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from composer.core.data_spec import DataSpec
from datasets import Dataset
from omegaconf import DictConfig
from scipy.sparse import csr_matrix
from streaming import Stream, StreamingDataLoader, StreamingDataset

from helical.models.tahoe.tahoe_x1.data import DataCollator
from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab


def build_streams(streams: dict[str, Any]) -> List[Stream]:
    """Builds a list of streams from a dictionary.

    Args:
        streams (dict[str, Any]): A dictionary of stream configurations.
    Returns:
        List[Stream]: A list of StreamingDataset.Stream objects.
    """
    return [Stream(**stream) for stream in streams.values()]


def build_dataloader(
    vocab: GeneVocab,
    loader_cfg: DictConfig,
    collator_cfg: DictConfig,
    device_batch_size: int,
) -> DataSpec:
    """Builds a dataloader from a config."""
    dataset_cfg = loader_cfg.dataset
    streams = dataset_cfg.get("streams")

    if streams:
        streams = build_streams(streams)
        remote, local = None, None
    else:
        remote, local = dataset_cfg.remote, dataset_cfg.local

    dataset = StreamingDataset(
        remote=remote,
        local=local,
        streams=streams,
        download_timeout=dataset_cfg.get("download_timeout", 300),
        allow_unsafe_types=dataset_cfg.get("allow_unsafe_types", True),
        shuffle=dataset_cfg.shuffle,
        predownload=dataset_cfg.get("predownload"),
        shuffle_seed=dataset_cfg.get("shuffle_seed"),
        num_canonical_nodes=dataset_cfg.get("num_canonical_nodes", 2),
        cache_limit=dataset_cfg.get("cache_limit"),
        batch_size=device_batch_size,
    )
    if isinstance(collator_cfg.mlm_probability, MutableSequence):
        mlm_probability = list(collator_cfg.mlm_probability)
    else:
        mlm_probability = collator_cfg.mlm_probability

    collate_fn = DataCollator(
        vocab=vocab,
        drug_to_id_path=collator_cfg.get("drug_to_id_path", None),
        do_padding=collator_cfg.get("do_padding", True),
        unexp_padding=loader_cfg.get("unexp_padding", False),
        pad_token_id=collator_cfg.pad_token_id,
        pad_value=collator_cfg.pad_value,
        do_mlm=collator_cfg.get("do_mlm", True),
        do_binning=collator_cfg.get("do_binning", True),
        log_transform=collator_cfg.get("log_transform", False),
        target_sum=collator_cfg.get("target_sum", 10000),
        mlm_probability=mlm_probability,
        mask_value=collator_cfg.mask_value,
        max_length=collator_cfg.max_length,
        sampling=collator_cfg.sampling,
        num_bins=collator_cfg.get("num_bins", 51),
        right_binning=collator_cfg.get("right_binning", False),
        keep_first_n_tokens=collator_cfg.get("keep_first_n_tokens", 1),
        use_chem_token=collator_cfg.get("use_chem_token", False),
    )

    data_loader = StreamingDataLoader(
        dataset,
        batch_size=device_batch_size,
        collate_fn=collate_fn,
        drop_last=loader_cfg.get("drop_last", False),
        num_workers=loader_cfg.get("num_workers", 8),
        pin_memory=loader_cfg.get("pin_memory", True),
        prefetch_factor=loader_cfg.get("prefetch_factor", 48),
        persistent_workers=loader_cfg.get("persistent_workers", True),
    )
    return DataSpec(dataloader=data_loader)


def build_perturbation_dataloader(
    loader_cfg: DictConfig,
    device_batch_size: int,
    isTrain: bool,
) -> DataSpec:
    """Builds a dataloader from a config for perturbation task.

    Args:
        loader_cfg (DictConfig): An omegaconf dictionary used to configure the loader.
        device_batch_size (int): The size of the batches (number of examples)
            that the dataloader will produce.
    """

    data_path = loader_cfg.get("dataset")["local"]
    max_len = loader_cfg.get("max_len")

    dataset = Dataset.load_from_disk(data_path)

    def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:

        genes = torch.stack([example["genes"] for example in examples])
        n_genes = len(genes[0])
        expressions_ctrls = torch.stack(
            [example["expressions_ctrl"] for example in examples],
        )
        expressions_perturbeds = torch.stack(
            [example["expressions_perturbed"] for example in examples],
        )
        perturb_flags = torch.stack([example["perturb_flag"] for example in examples])
        perturb_names = [example["perturb_name"] for example in examples]
        de_flags = torch.stack([example["de_flag"] for example in examples])

        # Randomly sample if sequence is longer than max_seq_len
        indices = (
            torch.randperm(n_genes)[:max_len] if isTrain else torch.arange(n_genes)
        )

        return {
            "genes": genes[:, indices],
            "expressions_ctrl": expressions_ctrls[:, indices],
            "expressions_perturbed": expressions_perturbeds[:, indices],
            "perturb_flags": perturb_flags[:, indices],
            "perturb_names": perturb_names,
            "de_flags": de_flags[:, indices],
        }

    data_loader = StreamingDataLoader(
        dataset,
        batch_size=device_batch_size,
        collate_fn=collate_fn,
    )

    return data_loader


class CountDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        count_matrix: Union[np.ndarray, csr_matrix],
        gene_ids: np.ndarray,
        add_cls_token: bool = True,
        cls_token_id: Optional[int] = None,
        pad_value: Optional[float] = None,
    ):
        """
        Args:
            count_matrix (np.ndarray or csr_matrix): A 2D expression count matrix of shape (n_cells, n_genes).
                If given as a dense NumPy array, it will be converted to sparse CSR format.
            gene_ids (np.ndarray): Integer Gene IDs corresponding to gene names in the count matrix (n_genes,).
            add_cls_token (bool, optional): If True, a CLS token will be added at the beginning of each sample.
                Defaults to True.
            cls_token_id (int, optional): The id of the <cls> token. Required if add_cls_token is True.
            pad_value (float, optional): The expression value used for PAD tokens.
                Required if add_cls_token is True.
        """
        if isinstance(count_matrix, np.ndarray):
            count_matrix = csr_matrix(count_matrix)
        if not isinstance(count_matrix, csr_matrix):
            raise ValueError(
                f"count_matrix must be either an np.ndarray or a scipy.sparse csr_matrix. found {type(count_matrix)}",
            )
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids

        self.add_cls_token = add_cls_token
        if self.add_cls_token:
            if cls_token_id is None or pad_value is None:
                raise ValueError(
                    "cls_token_id and pad_value must be provided when add_cls_token is True",
                )
            self.cls_token_id = cls_token_id
            self.pad_value = pad_value

    def __len__(self) -> int:
        return self.count_matrix.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.count_matrix.getrow(idx)
        nonzero_idx = row.indices
        values = row.data
        genes = self.gene_ids[nonzero_idx]
        if self.add_cls_token:
            genes = np.insert(genes, 0, self.cls_token_id)
            values = np.insert(values, 0, self.pad_value)
        return {
            "id": idx,
            "genes": torch.tensor(genes, dtype=torch.long),
            "expressions": torch.tensor(values, dtype=torch.float),
        }
