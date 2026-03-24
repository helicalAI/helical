from typing import List, Dict, Optional, Union, Tuple
import numpy as np
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
import torch
import anndata as ad
from scipy.sparse import issparse
import numba
import os
import json
from huggingface_hub import hf_hub_download
import pandas as pd

# Token IDs must match exactly with the original implementation
PAD_TOKEN = 0
MASK_TOKEN = 1
CLS_TOKEN = 2

# These mappings preserve the exact token IDs from the original implementation
MODALITY_DICT = {
    "dissociated": 3,
    "spatial": 4,
}

SPECIES_DICT = {
    "human": 5,
    "Homo sapiens": 5,
    "Mus musculus": 6,
    "mouse": 6,
}

TECHNOLOGY_DICT = {
    "merfish": 7,
    "MERFISH": 7,
    "cosmx": 8,
    "NanoString digital spatial profiling": 8,
    "Xenium": 9,
    "10x 5' v2": 10,
    "10x 3' v3": 11,
    "10x 3' v2": 12,
    "10x 5' v1": 13,
    "10x 3' v1": 14,
    "10x 3' transcription profiling": 15,
    "10x transcription profiling": 15,
    "10x 5' transcription profiling": 16,
    "CITE-seq": 17,
    "Smart-seq v4": 18,
}


def sf_normalize(X: np.ndarray) -> np.ndarray:
    """Size factor normalize to 10k counts."""
    X = X.copy()
    counts = np.array(X.sum(axis=1))
    # avoid zero division error
    counts += counts == 0.0
    # normalize to 10000 counts
    scaling_factor = 10000.0 / counts

    if issparse(X):
        from scipy.sparse import sparsefuncs

        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(
    x: np.ndarray, max_seq_len: int = -1, aux_tokens: int = 30
) -> np.ndarray:
    """Tokenize the input gene vector."""
    scores_final = np.empty(
        (x.shape[0], max_seq_len if max_seq_len > 0 else x.shape[1])
    )
    for i, cell in enumerate(x):
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices = sorted_indices + aux_tokens
        if max_seq_len:
            scores = np.zeros(max_seq_len, dtype=np.int32)
        else:
            scores = np.zeros_like(cell, dtype=np.int32)
        scores[: len(sorted_indices)] = sorted_indices.astype(np.int32)
        scores_final[i, :] = scores
    return scores_final


class NicheformerTokenizer(PreTrainedTokenizer):
    """Tokenizer for Nicheformer that handles single-cell data."""

    model_input_names = ["input_ids", "attention_mask"]
    vocab_files_names = {"vocab_file": "vocab.json"}

    modality_dict = MODALITY_DICT
    species_dict = SPECIES_DICT
    technology_dict = TECHNOLOGY_DICT

    def _load_reference_model(self):
        """Load reference model for gene alignment."""
        try:
            # Get the model name or path from the tokenizer
            repo_id = (
                self.name_or_path
                if hasattr(self, "name_or_path")
                else "aletlvl/Nicheformer"
            )

            # Download the reference model if not already cached
            model_path = hf_hub_download(repo_id=repo_id, filename="model.h5ad")
            return ad.read_h5ad(model_path)
        except Exception as e:
            print(f"Warning: Could not load reference model: {e}")
            return None

    def __init__(
        self,
        vocab_file=None,
        max_length: int = 1500,
        aux_tokens: int = 30,
        median_counts_per_gene: Optional[np.ndarray] = None,
        gene_names: Optional[List[str]] = None,
        technology_mean: Optional[Union[str, np.ndarray]] = None,
        **kwargs,
    ):
        # Initialize base vocabulary
        self._vocabulary = {
            "[PAD]": PAD_TOKEN,
            "[MASK]": MASK_TOKEN,
            "[CLS]": CLS_TOKEN,
        }

        if vocab_file is not None:
            with open(vocab_file, "r") as f:
                self._vocabulary.update(json.load(f))
        else:
            # Add modality tokens
            for name, idx in self.modality_dict.items():
                self._vocabulary[f"[MODALITY_{name}]"] = idx
            # Add species tokens
            for name, idx in self.species_dict.items():
                if name in ["Homo sapiens", "Mus musculus"]:
                    continue  # Skip redundant names
                self._vocabulary[f"[SPECIES_{name}]"] = idx
            # Add technology tokens
            for name, idx in self.technology_dict.items():
                if name in ["MERFISH", "10x transcription profiling"]:
                    continue  # Skip redundant names
                clean_name = name.lower().replace(" ", "_").replace("'", "_")
                self._vocabulary[f"[TECH_{clean_name}]"] = idx

            # Add gene tokens if provided
            if gene_names is not None:
                for i, gene in enumerate(gene_names):
                    self._vocabulary[gene] = i + aux_tokens
                # Save vocabulary
                os.makedirs("to_hf", exist_ok=True)
                with open("to_hf/vocab.json", "w") as f:
                    json.dump(self._vocabulary, f, indent=4)

        super().__init__(**kwargs)

        self.max_length = max_length
        self.aux_tokens = aux_tokens
        self.median_counts_per_gene = median_counts_per_gene
        self.gene_names = gene_names
        self.name_or_path = kwargs.get("name_or_path", "aletlvl/Nicheformer")

        # Set up special token mappings
        self._pad_token = "[PAD]"
        self._mask_token = "[MASK]"
        self._cls_token = "[CLS]"

        # Load technology mean if provided
        self.technology_mean = None
        if technology_mean is not None:
            self._load_technology_mean(technology_mean)

    def _load_technology_mean(self, technology_mean):
        """Load technology mean from file or array."""
        if isinstance(technology_mean, str):
            try:
                self.technology_mean = np.load(technology_mean)
                print(
                    f"Loaded technology mean from {technology_mean} with shape {self.technology_mean.shape}"
                )
            except Exception as e:
                print(
                    f"Warning: Could not load technology mean from {technology_mean}: {e}"
                )
        elif isinstance(technology_mean, np.ndarray):
            self.technology_mean = technology_mean
            print(
                f"Using provided technology mean array with shape {self.technology_mean.shape}"
            )
        else:
            print(f"Warning: Invalid technology_mean type: {type(technology_mean)}")

    def get_vocab(self) -> Dict[str, int]:
        """Returns the vocabulary mapping."""
        return self._vocabulary.copy()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text input."""
        # This tokenizer doesn't handle text input directly
        raise NotImplementedError("This tokenizer only works with gene expression data")

    def _convert_token_to_id(self, token: str) -> int:
        """Convert token to ID."""
        # First check special token mappings
        if token in self.modality_dict:
            return self.modality_dict[token]
        if token in self.species_dict:
            return self.species_dict[token]
        if token in self.technology_dict:
            return self.technology_dict[token]
        # Then check vocabulary
        return self._vocabulary.get(token, self._vocabulary["[PAD]"])

    def _convert_id_to_token(self, index: int) -> str:
        """Convert ID to token."""
        # First check special token mappings
        for token, idx in self.modality_dict.items():
            if idx == index:
                return token
        for token, idx in self.species_dict.items():
            if idx == index:
                return token
        for token, idx in self.technology_dict.items():
            if idx == index:
                return token
        # Then check vocabulary
        for token, idx in self._vocabulary.items():
            if idx == index:
                return token
        return "[PAD]"

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> Tuple[str]:
        """Save the vocabulary to a file."""
        vocab_file = os.path.join(
            save_directory,
            (filename_prefix + "-" if filename_prefix else "") + "vocab.json",
        )

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(self._vocabulary, f, ensure_ascii=False)

        return (vocab_file,)

    def _tokenize_gene_expression(self, x: np.ndarray) -> np.ndarray:
        """Tokenize gene expression matrix.

        Args:
            x: Gene expression matrix (cells x genes)

        Returns:
            Tokenized matrix
        """
        # Handle sparse input
        if issparse(x):
            x = x.toarray()

        # Normalize and scale
        x = np.nan_to_num(x)
        x = sf_normalize(x)
        if self.median_counts_per_gene is not None:
            median_counts = self.median_counts_per_gene.copy()
            median_counts += median_counts == 0
            x = x / median_counts.reshape((1, -1))

        # Apply technology mean normalization if available
        if (
            self.technology_mean is not None
            and self.technology_mean.shape[0] == x.shape[1]
        ):
            # Avoid division by zero
            safe_mean = np.maximum(self.technology_mean, 1e-6)
            x = x / safe_mean

        # Apply log1p transformation
        x = np.log1p(x)

        # Convert to tokens
        tokens = _sub_tokenize_data(x, self.max_length, self.aux_tokens)

        return tokens.astype(np.int32)

    def __call__(
        self, data: Union[ad.AnnData, np.ndarray], **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Tokenize gene expression data.

        Args:
            data: AnnData object or numpy array of gene expression data

        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        if isinstance(data, ad.AnnData):
            adata = data.copy()

            # Align with reference model if available
            if hasattr(self, "_load_reference_model"):
                reference_model = self._load_reference_model()
                if reference_model is not None:
                    # Store original column types before concatenation
                    original_types = {}
                    for col in ["modality", "specie", "assay"]:
                        if col in adata.obs.columns:
                            original_types[col] = adata.obs[col].dtype

                    # Concatenate and then remove the reference
                    adata = ad.concat([reference_model, adata], join="outer", axis=0)
                    adata = adata[1:]

                    # Restore original column types after concatenation
                    for col, dtype in original_types.items():
                        if col in adata.obs.columns:
                            try:
                                adata.obs[col] = adata.obs[col].astype(dtype)
                            except Exception as e:
                                print(
                                    f"Warning: Could not convert {col} back to {dtype}: {e}"
                                )

            # Get gene expression data
            X = adata.X

            # Get metadata for special tokens
            modality = (
                adata.obs["modality"] if "modality" in adata.obs.columns else None
            )
            species = adata.obs["specie"] if "specie" in adata.obs.columns else None
            technology = adata.obs["assay"] if "assay" in adata.obs.columns else None

            # Use integer values directly if available
            if modality is not None:
                try:
                    if pd.api.types.is_numeric_dtype(modality):
                        modality_tokens = modality.astype(int).tolist()
                    else:
                        modality_tokens = [
                            self.modality_dict.get(m, self._vocabulary["[PAD]"])
                            for m in modality
                        ]
                except Exception as e:
                    print(f"Warning: Error processing modality tokens: {e}")
                    modality_tokens = [self._vocabulary["[PAD]"]] * len(adata)
            else:
                modality_tokens = None

            if species is not None:
                try:
                    if pd.api.types.is_numeric_dtype(species):
                        species_tokens = species.astype(int).tolist()
                    else:
                        species_tokens = [
                            self.species_dict.get(s, self._vocabulary["[PAD]"])
                            for s in species
                        ]
                except Exception as e:
                    print(f"Warning: Error processing species tokens: {e}")
                    species_tokens = [self._vocabulary["[PAD]"]] * len(adata)
            else:
                species_tokens = None

            if technology is not None:
                try:
                    if pd.api.types.is_numeric_dtype(technology):
                        technology_tokens = technology.astype(int).tolist()
                    else:
                        technology_tokens = [
                            self.technology_dict.get(t, self._vocabulary["[PAD]"])
                            for t in technology
                        ]
                except Exception as e:
                    print(f"Warning: Error processing technology tokens: {e}")
                    technology_tokens = [self._vocabulary["[PAD]"]] * len(adata)
            else:
                technology_tokens = None
        else:
            X = data
            modality_tokens = None
            species_tokens = None
            technology_tokens = None

        # Tokenize gene expression data
        token_ids = self._tokenize_gene_expression(X)

        # Add special tokens if available - changed order to [species, technology, modality]
        special_tokens = np.zeros((token_ids.shape[0], 3), dtype=np.int64)
        special_token_mask = np.zeros((token_ids.shape[0], 3), dtype=bool)

        if species_tokens is not None:
            special_tokens[:, 0] = species_tokens
            special_token_mask[:, 0] = True

        if technology_tokens is not None:
            special_tokens[:, 1] = technology_tokens
            special_token_mask[:, 1] = True

        if modality_tokens is not None:
            special_tokens[:, 2] = modality_tokens
            special_token_mask[:, 2] = True

        # Only keep the special tokens that are present (have True in mask)
        special_tokens = special_tokens[:, special_token_mask[0]]

        if special_tokens.size > 0:
            token_ids = np.concatenate(
                [
                    special_tokens,
                    token_ids[:, : (self.max_length - special_tokens.shape[1])],
                ],
                axis=1,
            )

        # Create attention mask
        attention_mask = token_ids != self._vocabulary["[PAD]"]

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask),
        }

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        if self.gene_names is not None:
            return len(self.gene_names) + self.aux_tokens
        return (
            max(
                max(self.modality_dict.values()),
                max(self.species_dict.values()),
                max(self.technology_dict.values()),
            )
            + 1
        )

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a sequence of tokens to a string. Not used for gene expression."""
        raise NotImplementedError("This tokenizer only works with gene expression data")

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Build model inputs from a sequence by adding special tokens."""
        # For gene expression data, special tokens are handled in __call__
        return token_ids_0

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        """Get list where entries are [1] if a token is [special] else [0]."""
        # Consider tokens < aux_tokens as special
        return [1 if token_id < self.aux_tokens else 0 for token_id in token_ids_0]
