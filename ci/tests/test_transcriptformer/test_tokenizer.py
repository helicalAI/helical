import logging

import numpy as np
import pytest
import torch

from helical.models.transcriptformer.tokenizer.tokenizer import BatchGeneTokenizer

VOCAB = {"unknown": 0, "G1": 1, "G2": 2, "G3": 3}
GENES = np.array(["G3", "GX", "G1", "G2", "GY"])  # GX and GY are not in VOCAB


def _reference(names):
    """The previous implementation: one dict lookup per element."""
    return torch.tensor(np.vectorize(lambda x: VOCAB.get(x, VOCAB["unknown"]))(names))


@pytest.mark.parametrize(
    "index",
    [
        torch.tensor([[0, 2, 3], [4, 1, 0]]),  # unknowns in one row only
        torch.tensor([[2, 3, 0], [3, 0, 2]]),  # no unknowns
        torch.tensor([[1, 1, 1]]),  # repeats
    ],
)
def test_indexed_call_matches_lookup_per_element(index, caplog):
    tokenizer = BatchGeneTokenizer(VOCAB)
    expected = _reference(GENES[index.numpy()])

    with caplog.at_level(logging.WARNING):
        toks = tokenizer(GENES, index=index)

    assert toks.dtype == torch.long
    assert torch.equal(toks, expected)
    n_unknown = int((expected == 0).sum())
    warnings = [r.getMessage() for r in caplog.records]
    if n_unknown:
        assert warnings == [f"Warning: {n_unknown} genes not found in gene vocab"]
    else:
        assert warnings == []


@pytest.mark.parametrize("names", [GENES, GENES.reshape(1, 5)])
def test_call_without_index_matches_lookup_per_element(names):
    toks = BatchGeneTokenizer(VOCAB)(names)
    assert torch.equal(toks, _reference(names))
    assert toks.shape == names.shape


def test_process_batch_tokens_match_lookup_per_position():
    from helical.models.transcriptformer.data.dataloader import process_batch

    gene_names = np.array(["G1", "GX", "G2", "G3"])
    vocab = {**VOCAB, "[PAD]": 9}
    x = np.array([[5.0, 0.0, 2.0, 7.0], [0.0, 3.0, 4.0, 1.0]])

    result = process_batch(
        x,
        None,
        gene_names,
        BatchGeneTokenizer(vocab),
        None,
        sort_genes=True,
        randomize_order=False,
        max_len=3,
        pad_zeros=False,
        pad_token="[PAD]",
        gene_vocab=vocab,
        normalize_to_scale=None,
        clip_counts=None,
        aux_vocab=None,
    )

    ids = torch.argsort(torch.tensor(x, dtype=torch.float32), dim=1, descending=True)[
        :, :3
    ]
    expected = torch.tensor(
        [
            [vocab.get(g, vocab["unknown"]) for g in row]
            for row in gene_names[ids.numpy()]
        ]
    )
    assert torch.equal(result["gene_token_indices"], expected)
