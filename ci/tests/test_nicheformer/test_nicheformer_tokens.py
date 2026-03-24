"""
Tests for PAD / MASK token contract.

The invariant, confirmed by the embedding layer's padding_idx and masking.py:
    PAD_TOKEN  = 1   – excluded from attention, embedding always zero
    MASK_TOKEN = 0   – included in attention, model must predict the original
    CLS_TOKEN  = 2

These tests are pure unit tests: no disk access, no model downloads.
"""

import numpy as np
import torch
import pytest

from helical.models.nicheformer.tokenization_nicheformer import (
    PAD_TOKEN,
    MASK_TOKEN,
    CLS_TOKEN,
    _sub_tokenize_data,
)
import helical.models.nicheformer.masking as masking_module
from helical.models.nicheformer.masking import complete_masking
from helical.models.nicheformer.modeling_nicheformer import NicheformerModel
from helical.models.nicheformer.configuration_nicheformer import (
    NicheformerConfig as ModelConfig,
)


# ---------------------------------------------------------------------------
# 1. Token ID consistency across all three modules
# ---------------------------------------------------------------------------


class TestTokenIdConsistency:
    """All three modules must agree on the numeric value of each special token."""

    def test_pad_token_value(self):
        assert PAD_TOKEN == 1

    def test_mask_token_value(self):
        assert MASK_TOKEN == 0

    def test_cls_token_value(self):
        assert CLS_TOKEN == 2

    def test_pad_matches_masking_module(self):
        assert PAD_TOKEN == masking_module.PAD_TOKEN

    def test_mask_matches_masking_module(self):
        assert MASK_TOKEN == masking_module.MASK_TOKEN

    def test_cls_matches_masking_module(self):
        assert CLS_TOKEN == masking_module.CLS_TOKEN

    def test_embedding_padding_idx_matches_pad_token(self):
        """The embedding layer's padding_idx must equal PAD_TOKEN so that
        padding positions receive a zero embedding during both training and
        inference."""
        cfg = ModelConfig(n_tokens=100, context_length=50, learnable_pe=True)
        model = NicheformerModel(cfg)
        assert model.embeddings.padding_idx == PAD_TOKEN

    def test_pad_embedding_is_zero(self):
        """Consequence of padding_idx: the PAD row in the embedding table must
        be the zero vector."""
        cfg = ModelConfig(n_tokens=100, context_length=50, learnable_pe=True)
        model = NicheformerModel(cfg)
        pad_emb = model.embeddings(torch.tensor([PAD_TOKEN]))
        assert torch.all(pad_emb == 0), "PAD embedding is not zero"

    def test_mask_embedding_is_nonzero(self):
        """MASK is a learned token; its embedding must not be the zero vector
        after initialisation (xavier_normal_ will not produce exactly zeros)."""
        cfg = ModelConfig(n_tokens=100, context_length=50, learnable_pe=True)
        model = NicheformerModel(cfg)
        mask_emb = model.embeddings(torch.tensor([MASK_TOKEN]))
        assert not torch.all(mask_emb == 0), "MASK embedding is unexpectedly zero"


# ---------------------------------------------------------------------------
# 2. Tokenizer: trailing padding must carry PAD_TOKEN
# ---------------------------------------------------------------------------


class TestTokenizerPadding:
    """_sub_tokenize_data must fill unused trailing positions with PAD_TOKEN."""

    def _tokenize(self, x, max_seq_len=10, aux_tokens=30):
        return _sub_tokenize_data(
            x, max_seq_len=max_seq_len, aux_tokens=aux_tokens
        ).astype(np.int32)

    def test_sparse_cell_trailing_positions_are_pad(self):
        # 1 cell, 5 genes, only 2 non-zero → 8 trailing slots must be PAD
        x = np.zeros((1, 5), dtype=np.float32)
        x[0, 1] = 3.0
        x[0, 3] = 1.0
        tokens = self._tokenize(x)
        assert (tokens[0, 2:] == PAD_TOKEN).all()

    def test_all_zero_cell_is_all_pad(self):
        x = np.zeros((1, 5), dtype=np.float32)
        tokens = self._tokenize(x)
        assert (tokens[0] == PAD_TOKEN).all()

    def test_trailing_pad_is_not_mask_token(self):
        x = np.zeros((1, 5), dtype=np.float32)
        tokens = self._tokenize(x)
        assert not (tokens[0] == MASK_TOKEN).any()


# ---------------------------------------------------------------------------
# 3. Attention mask: PAD excluded, MASK included
# ---------------------------------------------------------------------------


class TestAttentionMask:
    """The attention mask derived from token IDs must treat PAD and MASK
    oppositely: PAD → False (excluded), MASK → True (included)."""

    def _make_mask(self, token_ids: np.ndarray) -> np.ndarray:
        # mirrors the tokenizer's attention_mask construction
        return token_ids != PAD_TOKEN

    def test_pad_positions_masked_out(self):
        ids = np.array([[30, 31, PAD_TOKEN, PAD_TOKEN]])
        mask = self._make_mask(ids)
        assert not mask[0, 2]
        assert not mask[0, 3]

    def test_gene_positions_attended(self):
        ids = np.array([[30, 31, PAD_TOKEN, PAD_TOKEN]])
        mask = self._make_mask(ids)
        assert mask[0, 0]
        assert mask[0, 1]

    def test_mask_token_is_attended(self):
        """A MASK token (value 0) placed in the sequence must be attended to,
        not silently treated as padding."""
        ids = np.array([[MASK_TOKEN, 30, PAD_TOKEN]])
        mask = self._make_mask(ids)
        assert mask[0, 0], "MASK token must not be excluded from attention"
        assert not mask[0, 2], "PAD token must be excluded from attention"


# ---------------------------------------------------------------------------
# 4. complete_masking: PAD never masked, MASK token used for substitution
# ---------------------------------------------------------------------------


class TestMaskingBehavior:
    """complete_masking must respect the PAD/MASK contract."""

    def _batch(self, seq):
        ids = torch.tensor([seq], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": ids != PAD_TOKEN}

    def test_pad_positions_never_replaced(self):
        seq = [30, PAD_TOKEN, 31, PAD_TOKEN, PAD_TOKEN]
        result = complete_masking(self._batch(seq), masking_p=1.0, n_tokens=100)
        for i, tok in enumerate(seq):
            if tok == PAD_TOKEN:
                assert result["masked_indices"][0, i].item() == PAD_TOKEN

    def test_masked_positions_are_not_pad(self):
        """Replacing a real token must never produce PAD_TOKEN (1)."""
        seq = list(range(30, 50))  # 20 gene tokens, no padding
        torch.manual_seed(0)
        result = complete_masking(self._batch(seq), masking_p=1.0, n_tokens=200)
        masked_pos = result["mask"][0]
        for i in range(len(seq)):
            if masked_pos[i]:
                val = result["masked_indices"][0, i].item()
                assert val != PAD_TOKEN, f"Position {i} was masked to PAD_TOKEN"

    def test_random_replacement_is_not_a_noop(self):
        """The 10 % random-replacement branch must actually write to the tensor.
        With a long sequence and p=1.0, ~10 % of tokens become random gene
        tokens (≠ original and ≠ MASK_TOKEN).  A no-op bug produces zero such
        positions."""
        seq = list(range(30, 130))  # 100 gene tokens
        torch.manual_seed(42)
        result = complete_masking(self._batch(seq), masking_p=1.0, n_tokens=200)
        original = result["input_ids"][0]
        modified = result["masked_indices"][0]
        randomly_replaced = (
            (modified != original) & (modified != MASK_TOKEN) & (modified != PAD_TOKEN)
        )
        assert (
            randomly_replaced.any()
        ), "No random token replacements found — the assignment is likely a no-op"
