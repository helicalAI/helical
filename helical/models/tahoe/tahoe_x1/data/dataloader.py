# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
from typing import Dict, Optional, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix


class CountDataset(torch.utils.data.Dataset):
    """Dataset for loading sparse count matrices as PyTorch tensors.

    Used for inference on single-cell RNA-seq data stored in sparse CSR format.
    """

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
