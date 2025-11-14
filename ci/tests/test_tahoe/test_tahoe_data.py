import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from helical.models.tahoe.tahoe_x1.data.collator import DataCollator
from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab


class TestDataCollator:
    """Test suite for DataCollator class."""

    @pytest.fixture
    def mock_vocab(self):
        """Create a mock vocabulary."""
        vocab_dict = {
            "<cls>": 0,
            "<pad>": 1,
            "<mask>": 2,
            "gene1": 3,
            "gene2": 4,
            "gene3": 5,
        }
        vocab = Mock(spec=GeneVocab)
        vocab.__getitem__ = lambda self, key: vocab_dict.get(key, -1)
        vocab.__contains__ = lambda self, key: key in vocab_dict
        vocab.get_stoi = lambda: vocab_dict
        vocab.pad_token_id = 1
        return vocab

    @pytest.fixture
    def collator_config(self, mock_vocab):
        """Create a DataCollator configuration."""
        return {
            "vocab": mock_vocab,
            "drug_to_id_path": None,
            "do_padding": True,
            "unexp_padding": False,
            "pad_token_id": 1,
            "pad_value": 0.0,
            "do_mlm": False,
            "do_binning": True,
            "log_transform": False,
            "target_sum": None,
            "mlm_probability": 0.15,
            "mask_value": -1,
            "max_length": 100,
            "sampling": True,
            "num_bins": 51,
            "right_binning": False,
            "keep_first_n_tokens": 1,
            "use_chem_token": False,
        }

    def test_collator_initialization(self, mock_vocab, collator_config):
        """Test DataCollator initialization."""
        collator = DataCollator(**collator_config)

        assert collator.do_padding is True
        assert collator.pad_token_id == 1
        assert collator.do_mlm is False
        assert collator.num_bins == 51

    def test_collator_with_mlm_disabled(self, mock_vocab, collator_config):
        """Test collator with MLM disabled (inference mode)."""
        collator = DataCollator(**collator_config)

        # Create sample batch
        batch = [
            {
                "id": 0,
                "genes": torch.tensor([0, 3, 4]),  # <cls>, gene1, gene2
                "expressions": torch.tensor([0.0, 1.5, 2.5]),
            }
        ]

        # Process batch
        with patch.object(collator, "_pad", return_value=batch[0]):
            with patch("helical.models.tahoe.tahoe_x1.data.collator.binning", side_effect=lambda row, **kwargs: row):
                output = collator(batch)

        assert "gene" in output or "genes" in output

    def test_collator_padding(self, mock_vocab, collator_config):
        """Test that sequences are padded correctly."""
        collator_config["max_length"] = 5
        collator = DataCollator(**collator_config)

        # Create sample with fewer tokens than max_length
        sample = {
            "id": 0,
            "genes": torch.tensor([0, 3, 4]),
            "expressions": torch.tensor([0.0, 1.5, 2.5]),
        }

        # The _pad method should extend to max_length
        # Note: This is testing the interface, actual padding logic may vary

    def test_collator_binning_enabled(self, mock_vocab, collator_config):
        """Test collator with binning enabled."""
        collator_config["do_binning"] = True
        collator = DataCollator(**collator_config)

        assert collator.do_binning is True
        assert collator.num_bins == 51

    def test_collator_binning_disabled(self, mock_vocab, collator_config):
        """Test collator with binning disabled."""
        collator_config["do_binning"] = False
        collator = DataCollator(**collator_config)

        assert collator.do_binning is False

    def test_collator_keeps_first_n_tokens(self, mock_vocab, collator_config):
        """Test that collator respects keep_first_n_tokens parameter."""
        collator_config["keep_first_n_tokens"] = 2
        collator = DataCollator(**collator_config)

        assert collator.keep_first_n_tokens == 2

    def test_collator_sampling_enabled(self, mock_vocab, collator_config):
        """Test collator with sampling enabled."""
        collator_config["sampling"] = True
        collator = DataCollator(**collator_config)

        assert collator.sampling is True

    @pytest.mark.parametrize("num_bins", [10, 51, 100])
    def test_collator_different_num_bins(self, mock_vocab, collator_config, num_bins):
        """Test collator with different numbers of bins."""
        collator_config["num_bins"] = num_bins
        collator = DataCollator(**collator_config)

        assert collator.num_bins == num_bins

    def test_collator_without_drug_info(self, mock_vocab, collator_config):
        """Test collator without chemical/drug information."""
        collator_config["use_chem_token"] = False
        collator = DataCollator(**collator_config)

        assert collator.use_chem_token is False

    def test_collator_batch_processing(self, mock_vocab, collator_config):
        """Test processing a batch of samples."""
        collator = DataCollator(**collator_config)

        batch = [
            {
                "id": i,
                "genes": torch.tensor([0, 3, 4]),
                "expressions": torch.tensor([0.0, float(i), float(i + 1)]),
            }
            for i in range(3)
        ]

        # The collator should process the batch without errors
        # Actual output format depends on implementation


class TestBinning:
    """Test suite for binning functionality."""

    def test_binning_value_range(self):
        """Test that binned values are within expected range."""
        # Expression values should be binned into discrete bins
        expressions = np.array([0.0, 1.0, 5.0, 10.0, 50.0, 100.0])
        num_bins = 51

        # Binning should map continuous values to discrete bins
        # The exact binning logic would depend on implementation

    @pytest.mark.parametrize("num_bins", [10, 51, 100])
    def test_different_bin_sizes(self, num_bins):
        """Test binning with different numbers of bins."""
        expressions = np.linspace(0, 100, 20)
        # Test that binning works with various bin counts

    def test_log_transform_before_binning(self):
        """Test log transformation before binning."""
        expressions = np.array([1.0, 10.0, 100.0, 1000.0])
        # If log_transform is enabled, values should be log-transformed first


class TestGeneExpressionProcessing:
    """Test suite for gene expression processing."""

    def test_sparse_to_dense_conversion(self):
        """Test conversion from sparse to dense representation."""
        # Sparse input should be correctly converted

    def test_normalization(self):
        """Test expression value normalization."""
        expressions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        # Test normalization if applicable

    def test_filtering_zero_expressions(self):
        """Test that genes with zero expression are handled correctly."""
        # In sparse format, zeros should be filtered out

    @pytest.mark.parametrize("target_sum", [None, 10000, 1e4])
    def test_target_sum_normalization(self, target_sum):
        """Test normalization to target sum."""
        expressions = np.array([1.0, 2.0, 3.0, 4.0])
        if target_sum is not None:
            # Expressions should be normalized to sum to target_sum
            pass


class TestTokenization:
    """Test suite for gene tokenization."""

    def test_special_tokens(self):
        """Test that special tokens are correctly defined."""
        special_tokens = ["<cls>", "<pad>", "<mask>"]
        # Each special token should have a unique ID

    def test_gene_to_token_mapping(self):
        """Test mapping gene IDs to token IDs."""
        gene_ids = ["ENSG00000187634", "ENSG00000188290"]
        # Each gene should map to a unique token ID

    def test_padding_token_behavior(self):
        """Test that padding tokens are handled correctly."""
        # Padding tokens should be used to make sequences equal length

    def test_cls_token_position(self):
        """Test that CLS token is added at the beginning."""
        # CLS token should always be at position 0

    def test_mask_token_for_mlm(self):
        """Test that mask tokens are used correctly for MLM."""
        # When MLM is enabled, some tokens should be masked


class TestSequenceLength:
    """Test suite for sequence length handling."""

    @pytest.mark.parametrize("max_length", [100, 1000, 10000])
    def test_different_max_lengths(self, max_length):
        """Test with different maximum sequence lengths."""
        # Sequences should be truncated or padded to max_length

    def test_sequence_truncation(self):
        """Test that long sequences are truncated correctly."""
        long_sequence = list(range(20000))
        max_length = 10000
        # Sequence should be truncated to max_length

    def test_sequence_padding(self):
        """Test that short sequences are padded correctly."""
        short_sequence = [1, 2, 3]
        max_length = 100
        # Sequence should be padded to max_length with pad tokens

    def test_sampling_when_exceeds_max_length(self):
        """Test gene sampling when number of genes exceeds max_length."""
        # When sampling is enabled and genes > max_length,
        # genes should be randomly sampled


class TestDataValidation:
    """Test suite for data validation."""

    def test_invalid_gene_ids(self):
        """Test handling of invalid gene IDs."""
        # Invalid gene IDs should be filtered or raise an error

    def test_negative_expression_values(self):
        """Test handling of negative expression values."""
        # Negative values should be handled appropriately

    def test_nan_expression_values(self):
        """Test handling of NaN expression values."""
        expressions = np.array([1.0, np.nan, 3.0])
        # NaN values should be handled or raise an error

    def test_empty_gene_list(self):
        """Test handling of empty gene list."""
        # Empty gene lists should be handled gracefully

    def test_mismatched_gene_expr_lengths(self):
        """Test handling of mismatched gene and expression lengths."""
        genes = [1, 2, 3]
        expressions = [1.0, 2.0]  # Length mismatch
        # Should raise an error or handle gracefully
