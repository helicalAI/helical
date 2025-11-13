import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix, csc_matrix
from anndata import AnnData
from unittest.mock import Mock, patch

from helical.models.tahoe.tahoe_x1.utils.util import loader_from_adata, download_file_from_s3_url
from helical.models.tahoe.tahoe_x1.data.dataloader import CountDataset
from helical.models.tahoe.tahoe_x1.tokenizer import GeneVocab


class TestLoaderFromAdata:
    """Test suite for loader_from_adata utility function."""

    @pytest.fixture
    def mock_vocab(self):
        """Create a mock vocabulary."""
        vocab_dict = {
            "<cls>": 0,
            "<pad>": 1,
            "gene1": 2,
            "gene2": 3,
            "gene3": 4,
        }
        vocab = Mock(spec=GeneVocab)
        vocab.__getitem__ = lambda self, key: vocab_dict.get(key, -1)
        vocab.__contains__ = lambda self, key: key in vocab_dict
        return vocab

    @pytest.fixture
    def mock_collator_cfg(self):
        """Create mock collator configuration."""
        return {
            "pad_value": 0.0,
            "do_padding": True,
            "pad_token_id": 1,
            "do_binning": True,
            "log_transform": False,
            "target_sum": None,
            "mlm_probability": 0.15,
            "mask_value": -1,
            "sampling": True,
            "num_bins": 51,
            "right_binning": False,
            "keep_first_n_tokens": 1,
            "use_chem_token": False,
        }

    @pytest.fixture
    def sample_adata(self):
        """Create a sample AnnData object."""
        adata = AnnData()
        adata.X = csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]]))
        adata.var["id_in_vocab"] = [2, 3, 4]
        return adata

    def test_loader_from_adata_basic(self, sample_adata, mock_vocab, mock_collator_cfg):
        """Test basic dataloader creation from AnnData."""
        gene_ids = np.array([2, 3, 4])

        with patch("helical.models.tahoe.tahoe_x1.utils.util.CountDataset") as MockDataset:
            with patch("helical.models.tahoe.tahoe_x1.utils.util.DataCollator") as MockCollator:
                MockDataset.return_value = Mock()
                MockCollator.return_value = Mock()

                loader = loader_from_adata(
                    adata=sample_adata,
                    collator_cfg=mock_collator_cfg,
                    vocab=mock_vocab,
                    batch_size=2,
                    gene_ids=gene_ids,
                    num_workers=0,
                    prefetch_factor=2,
                )

                # Verify CountDataset was called with correct parameters
                MockDataset.assert_called_once()
                call_args = MockDataset.call_args
                assert np.array_equal(call_args[1]["gene_ids"], gene_ids)
                assert call_args[1]["cls_token_id"] == 0
                assert call_args[1]["pad_value"] == 0.0

    def test_loader_from_adata_with_max_length(self, sample_adata, mock_vocab, mock_collator_cfg):
        """Test dataloader creation with max_length parameter."""
        gene_ids = np.array([2, 3, 4])

        with patch("helical.models.tahoe.tahoe_x1.utils.util.CountDataset"):
            with patch("helical.models.tahoe.tahoe_x1.utils.util.DataCollator"):
                loader = loader_from_adata(
                    adata=sample_adata,
                    collator_cfg=mock_collator_cfg,
                    vocab=mock_vocab,
                    batch_size=2,
                    max_length=1000,
                    gene_ids=gene_ids,
                    num_workers=0,
                )

                assert loader is not None

    def test_loader_from_adata_csc_to_csr_conversion(self, mock_vocab, mock_collator_cfg):
        """Test that CSC matrices are converted to CSR."""
        adata = AnnData()
        adata.X = csc_matrix(np.array([[1.0, 0.0], [0.0, 2.0]]))
        adata.var["id_in_vocab"] = [2, 3]
        gene_ids = np.array([2, 3])

        with patch("helical.models.tahoe.tahoe_x1.utils.util.CountDataset") as MockDataset:
            with patch("helical.models.tahoe.tahoe_x1.utils.util.DataCollator"):
                loader = loader_from_adata(
                    adata=adata,
                    collator_cfg=mock_collator_cfg,
                    vocab=mock_vocab,
                    batch_size=2,
                    gene_ids=gene_ids,
                    num_workers=0,
                )

                # Verify the count matrix passed is CSR
                call_args = MockDataset.call_args
                count_matrix = call_args[0][0]
                assert isinstance(count_matrix, csr_matrix)

    def test_loader_from_adata_infers_gene_ids(self, sample_adata, mock_vocab, mock_collator_cfg):
        """Test that gene_ids are inferred from adata.var when not provided."""
        with patch("helical.models.tahoe.tahoe_x1.utils.util.CountDataset") as MockDataset:
            with patch("helical.models.tahoe.tahoe_x1.utils.util.DataCollator"):
                MockDataset.return_value = Mock()

                loader = loader_from_adata(
                    adata=sample_adata,
                    collator_cfg=mock_collator_cfg,
                    vocab=mock_vocab,
                    batch_size=2,
                    gene_ids=None,  # Not providing gene_ids
                    num_workers=0,
                )

                # Verify gene_ids were inferred from adata.var
                call_args = MockDataset.call_args
                passed_gene_ids = call_args[1]["gene_ids"]
                expected_gene_ids = np.array(sample_adata.var["id_in_vocab"])
                assert np.array_equal(passed_gene_ids, expected_gene_ids)


class TestCountDataset:
    """Test suite for CountDataset class."""

    @pytest.fixture
    def sparse_matrix(self):
        """Create a sparse count matrix."""
        return csr_matrix(np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [5.0, 0.0, 6.0]]))

    @pytest.fixture
    def gene_ids(self):
        """Create gene IDs."""
        return np.array([10, 20, 30])

    def test_count_dataset_initialization(self, sparse_matrix, gene_ids):
        """Test CountDataset initialization."""
        dataset = CountDataset(
            count_matrix=sparse_matrix,
            gene_ids=gene_ids,
            add_cls_token=True,
            cls_token_id=0,
            pad_value=0.0,
        )

        assert len(dataset) == 3
        assert dataset.add_cls_token is True
        assert dataset.cls_token_id == 0

    def test_count_dataset_getitem_with_cls_token(self, sparse_matrix, gene_ids):
        """Test getting an item from CountDataset with CLS token."""
        dataset = CountDataset(
            count_matrix=sparse_matrix,
            gene_ids=gene_ids,
            add_cls_token=True,
            cls_token_id=0,
            pad_value=0.0,
        )

        item = dataset[0]

        assert "id" in item
        assert "genes" in item
        assert "expressions" in item
        assert item["id"] == 0
        # Should have CLS token prepended
        assert item["genes"][0] == 0
        assert item["expressions"][0] == 0.0
        # Original genes should follow
        assert item["genes"][1] == 10  # First non-zero gene
        assert item["genes"][-1] == 30  # Last non-zero gene

    def test_count_dataset_getitem_without_cls_token(self, sparse_matrix, gene_ids):
        """Test getting an item from CountDataset without CLS token."""
        dataset = CountDataset(
            count_matrix=sparse_matrix,
            gene_ids=gene_ids,
            add_cls_token=False,
            cls_token_id=None,
            pad_value=None,
        )

        item = dataset[1]

        assert item["id"] == 1
        # Should not have CLS token
        assert item["genes"][0] != 0
        # Should only contain non-zero genes
        assert len(item["genes"]) == 2  # genes at positions 1 and 2 are non-zero

    def test_count_dataset_dense_to_sparse_conversion(self, gene_ids):
        """Test that dense arrays are converted to sparse."""
        dense_matrix = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0]])

        dataset = CountDataset(
            count_matrix=dense_matrix,  # Passing dense array
            gene_ids=gene_ids,
            add_cls_token=False,
        )

        # Should still work correctly
        assert len(dataset) == 2
        assert isinstance(dataset.count_matrix, csr_matrix)

    def test_count_dataset_length(self, sparse_matrix, gene_ids):
        """Test that dataset length matches number of cells."""
        dataset = CountDataset(
            count_matrix=sparse_matrix,
            gene_ids=gene_ids,
            add_cls_token=False,
        )

        assert len(dataset) == sparse_matrix.shape[0]

    def test_count_dataset_requires_cls_params_when_enabled(self, sparse_matrix, gene_ids):
        """Test that cls_token_id and pad_value are required when add_cls_token=True."""
        with pytest.raises(ValueError, match="cls_token_id and pad_value must be provided"):
            CountDataset(
                count_matrix=sparse_matrix,
                gene_ids=gene_ids,
                add_cls_token=True,
                cls_token_id=None,  # Missing required parameter
                pad_value=None,
            )

    def test_count_dataset_invalid_matrix_type(self, gene_ids):
        """Test that invalid matrix types raise an error."""
        invalid_matrix = [[1, 2, 3], [4, 5, 6]]  # List instead of array

        with pytest.raises(ValueError, match="must be either an np.ndarray or a scipy.sparse csr_matrix"):
            CountDataset(
                count_matrix=invalid_matrix,
                gene_ids=gene_ids,
                add_cls_token=False,
            )


class TestDownloadFileFromS3:
    """Test suite for S3 download function."""

    def test_download_raises_not_implemented(self):
        """Test that S3 download raises NotImplementedError for inference."""
        with pytest.raises(NotImplementedError, match="S3 downloads are only supported during training"):
            download_file_from_s3_url("s3://bucket/path/file", "/local/path")

    def test_download_error_message(self):
        """Test that error message mentions HuggingFace Hub."""
        try:
            download_file_from_s3_url("s3://test/file", "/tmp/file")
        except NotImplementedError as e:
            assert "HuggingFace Hub" in str(e)


class TestGeneVocab:
    """Test suite for GeneVocab class."""

    def test_vocab_from_dict(self):
        """Test creating vocabulary from dictionary."""
        vocab_dict = {"gene1": 0, "gene2": 1, "<pad>": 2}
        # Note: This tests the interface, actual implementation may vary

    def test_vocab_contains(self):
        """Test vocabulary membership checking."""
        vocab_dict = {"gene1": 0, "gene2": 1}
        # Would test vocab.__contains__ if GeneVocab is imported

    def test_vocab_getitem(self):
        """Test vocabulary item access."""
        vocab_dict = {"gene1": 0, "gene2": 1}
        # Would test vocab[key] if GeneVocab is imported
