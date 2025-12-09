# Copyright (C) Tahoe Therapeutics 2025. All rights reserved.
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

class GeneVocab:
    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], Dict[str, int], "GeneVocab"],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        """Initialize the vocabulary.

        Args:
            gene_list_or_vocab: Either a list of gene names, a dict mapping tokens to indices,
                                or an existing GeneVocab instance.
            specials: Optional list of special tokens to include.
                     (When initializing from an existing GeneVocab, this must be None.)
            special_first: Whether to add special tokens at the beginning.
            default_token: The default token (typically used for padding).
        """
        if isinstance(gene_list_or_vocab, GeneVocab):
            if specials is not None:
                raise ValueError(
                    "Cannot provide specials when initializing from an existing GeneVocab.",
                )
            # Copy the internal mappings from the provided GeneVocab.
            self.token_to_index = gene_list_or_vocab.token_to_index.copy()
            self.index_to_token = gene_list_or_vocab.index_to_token.copy()
        elif isinstance(gene_list_or_vocab, dict):
            # Initialize directly from a token-to-index dictionary.
            self.token_to_index = gene_list_or_vocab.copy()
            self.index_to_token = {
                idx: token for token, idx in gene_list_or_vocab.items()
            }
        elif isinstance(gene_list_or_vocab, list):
            # Build the vocabulary from a list of tokens.
            self.token_to_index = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
            self.index_to_token = {
                idx: token for token, idx in self.token_to_index.items()
            }
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list, a dict, or a GeneVocab instance.",
            )

        self.default_index: Optional[int] = None
        self._pad_token: Optional[str] = None
        if default_token is not None and default_token in self.token_to_index:
            self.set_default_token(default_token)

    def _build_vocab_from_iterator(
        self,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> Dict[str, int]:
        """Build a token-to-index mapping from an iterator of tokens.

        Args:
            iterator: An iterable yielding tokens.
            min_freq: Only include tokens whose frequency is at least min_freq.
            specials: Optional list of special tokens.
            special_first: Whether to place special tokens at the beginning.

        Returns:
            A dictionary mapping tokens to indices.
        """
        counter = Counter(iterator)
        # Remove special tokens if they appear in the counter.
        if specials is not None:
            for tok in specials:
                counter.pop(tok, None)
        # Filter tokens by min frequency.
        filtered = [
            (token, freq) for token, freq in counter.items() if freq >= min_freq
        ]
        # Sort tokens by frequency (descending) then lexicographically (ascending).
        filtered.sort(key=lambda x: (-x[1], x[0]))
        tokens = [token for token, freq in filtered]
        if specials is not None:
            tokens = specials + tokens if special_first else tokens + specials
        return {token: idx for idx, token in enumerate(tokens)}

    def __contains__(self, token: str) -> bool:
        return token in self.token_to_index

    def __getitem__(self, token: str) -> int:
        """Get the index for a given token.

        If the token is not found, return the default index if set; otherwise,
        raise a KeyError.
        """
        if token in self.token_to_index:
            return self.token_to_index[token]
        elif self.default_index is not None:
            return self.default_index
        else:
            raise KeyError(f"Token {token} not found in vocabulary.")

    def insert_token(self, token: str, index: int) -> None:
        """Insert a token at a specific index.

        Tokens already present are left unchanged; otherwise, the vocabulary is
        re-indexed.
        """
        if token in self.token_to_index:
            return
        # Build a list of tokens in current order.
        tokens = [self.index_to_token[i] for i in range(len(self.token_to_index))]
        tokens.insert(index, token)
        # Rebuild the mappings.
        self.token_to_index = {tok: idx for idx, tok in enumerate(tokens)}
        self.index_to_token = {idx: tok for idx, tok in enumerate(tokens)}  # noqa:C416

    def append_token(self, token: str) -> None:
        """Append a token to the end of the vocabulary."""
        if token in self.token_to_index:
            return
        index = len(self.token_to_index)
        self.token_to_index[token] = index
        self.index_to_token[index] = token

    def set_default_token(self, token: str) -> None:
        """Set the default token.

        This token index will be used for unknown tokens.
        """
        if token not in self.token_to_index:
            raise ValueError(f"Default token '{token}' is not in the vocabulary.")
        self.default_index = self.token_to_index[token]
        self._pad_token = token

    @property
    def pad_token(self) -> Optional[str]:
        """Get the pad token."""
        return self._pad_token

    @pad_token.setter
    def pad_token(self, token: str) -> None:
        """Set the pad token (must already be in the vocabulary)."""
        if token not in self.token_to_index:
            raise ValueError(f"Pad token '{token}' is not in the vocabulary.")
        self._pad_token = token

    def save_json(self, file_path: Union[Path, str]) -> None:
        """Save the vocabulary (the token-to-index mapping) to a JSON file."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.token_to_index, f, indent=2)

    def get_stoi(self) -> Dict[str, int]:
        """Return a copy of the token-to-index mapping."""
        return self.token_to_index.copy()

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "GeneVocab":
        """Create a GeneVocab from a file.

        Supported file types: .pkl and .json.
        """
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
            return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
            return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. Only .pkl and .json are supported.",
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> "GeneVocab":
        """Create a GeneVocab from a dictionary mapping tokens to indices."""
        instance = cls(token2idx)
        if default_token is not None and default_token in instance.token_to_index:
            instance.set_default_token(default_token)
        return instance

    def __len__(self) -> int:
        return len(self.token_to_index)
