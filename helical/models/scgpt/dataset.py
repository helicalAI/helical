import os
from typing import Union
import numpy as np
import torch
from scipy import sparse

PathLike = Union[str, os.PathLike]


class Dataset(torch.utils.data.Dataset):
    def __init__(self, count_matrix, gene_ids, vocab, model_configs, batch_ids=None):
        self.count_matrix = count_matrix
        self.gene_ids = gene_ids
        self.batch_ids = batch_ids
        self.vocab = vocab
        self.model_configs = model_configs
        # Track if the count_matrix is sparse for efficient access
        self.is_sparse = sparse.issparse(count_matrix)

    def __len__(self):
        # Handle both sparse and dense matrices
        if not hasattr(self, 'count_matrix'):
            raise AttributeError("Dataset is missing count_matrix attribute")
        if self.count_matrix is None:
            raise ValueError("Dataset count_matrix is None")

        if self.is_sparse:
            return self.count_matrix.shape[0]
        return len(self.count_matrix)


    def __getstate__(self):
        """Custom pickle serialization to handle sparse matrices."""
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization to handle sparse matrices."""
        self.__dict__.update(state)
        # Re-check if count_matrix is sparse after unpickling
        self.is_sparse = sparse.issparse(self.count_matrix)

    def __getitem__(self, idx):
        # Get the row - convert to dense only if sparse
        row = self.count_matrix[idx]
        if self.is_sparse:
            # Convert single row from sparse to dense (memory efficient)
            # Handle both 1D and 2D sparse row outputs
            if hasattr(row, 'toarray'):
                row = row.toarray().flatten()
            else:
                row = np.asarray(row).flatten()

        # Ensure row is a 1D numpy array
        if not isinstance(row, np.ndarray):
            row = np.asarray(row).flatten()

        # Find non-zero elements
        nonzero_idx = np.nonzero(row)[0]
        values = row[nonzero_idx]
        genes = self.gene_ids[nonzero_idx]

        # append <cls> token at the beginning
        genes = np.insert(genes, 0, self.vocab["<cls>"])
        values = np.insert(values, 0, self.model_configs["pad_value"])
        genes = torch.from_numpy(genes).long()
        values = torch.from_numpy(values).float()
        output = {
            "id": idx,
            "genes": genes,
            "expressions": values,
        }
        if self.batch_ids is not None:
            output["batch_labels"] = self.batch_ids[idx]
        return output
