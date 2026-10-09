import logging
import numpy as np
import torch


class BatchGeneTokenizer:
    def __init__(self, gene_vocab):
        self.gene_vocab = gene_vocab
        self.unknown_token = gene_vocab["unknown"]

    def gene_map(self, x):
        return self.gene_vocab.get(x, self.unknown_token)

    def __call__(self, gene_names, index=None):
        """Map gene names to tokens.

        With ``index``, return ``tokens[index]``: the names are looked up once
        each and the result is gathered, instead of one dict lookup per
        position of a cells x genes grid.
        """
        toks = torch.tensor(
            [self.gene_map(x) for x in np.ravel(gene_names)], dtype=torch.long
        ).reshape(np.shape(gene_names))
        if index is not None:
            toks = toks[index]
        unknown_mask = toks == self.unknown_token
        if unknown_mask.any():
            n_unknown = unknown_mask.sum().item()
            logging.warning(f"Warning: {n_unknown} genes not found in gene vocab")
        return toks


class BatchObsTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, obs):
        return torch.tensor(
            [
                self.vocab[key].get(obs[key], self.vocab[key]["unknown"])
                for key in self.vocab
            ],
            dtype=torch.int64,
        )
