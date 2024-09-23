import pytest
import numpy as np
from unittest import mock
from anndata import AnnData
from pathlib import Path
import scipy.sparse as sp
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer, sum_ensembl_ids
from helical.constants.paths import CACHE_DIR_HELICAL

class TestGeneformerTokenizer:
    model_dir_v1 = Path(CACHE_DIR_HELICAL, "geneformer", "v1")
    model_dir_v2 = Path(CACHE_DIR_HELICAL, "geneformer", "v2")
    
    files_config_v1 = {
        "gene_median_path": model_dir_v1 / "gene_median_dictionary.pkl",
        "token_path": model_dir_v1 / "token_dictionary.pkl",
        "gene_mapping_path": model_dir_v1 / "ensembl_mapping_dict.pkl",
    }
    
    files_config_v2 = {
        "gene_median_path": model_dir_v2 / "gene_median_dictionary.pkl",
        "token_path": model_dir_v2 / "token_dictionary.pkl",
        "gene_mapping_path": model_dir_v2 / "ensembl_mapping_dict.pkl",
    }

    @pytest.fixture
    def tokenizer_v1(self):
        return TranscriptomeTokenizer(
            gene_median_file=self.files_config_v1["gene_median_path"],
            token_dictionary_file=self.files_config_v1["token_path"],
            gene_mapping_file=self.files_config_v1["gene_mapping_path"],
            model_input_size=2048,
            special_token=False,
        )

    @pytest.fixture
    def tokenizer_v2(self):
        return TranscriptomeTokenizer(
            gene_median_file=self.files_config_v2["gene_median_path"],
            token_dictionary_file=self.files_config_v2["token_path"],
            gene_mapping_file=self.files_config_v2["gene_mapping_path"],
            model_input_size=4096,
            special_token=True,
        )

    def test_tokenizer_initialization(self, tokenizer_v1, tokenizer_v2):
        assert tokenizer_v1.model_input_size == 2048
        assert not tokenizer_v1.special_token
        assert tokenizer_v2.model_input_size == 4096
        assert tokenizer_v2.special_token

    @pytest.mark.parametrize("collapse_gene_ids, expected_shape, expected_exception, data_format", [
    (True, (2, 2), None, 'h5ad'),
    (False, (2, 3), ValueError, 'h5ad'),
    (True, (2, 3), ValueError, 'bad_format')
    ])
    def test_sum_ensembl_ids_h5ad(self, collapse_gene_ids, expected_shape, expected_exception, data_format):
        # Create a test AnnData object
        adata = AnnData(X=sp.csr_matrix([[1, 2, 3], [4, 5, 6]]))
        adata.var['ensembl_id'] = ['ENSG1', 'ENSG2', 'ENSG2']
        adata.obs['n_counts'] = [6, 15]

        # Mock gene_mapping_dict and gene_token_dict
        gene_mapping_dict = {'ENSG1': 'ENSG1', 'ENSG2': 'ENSG2'}
        gene_token_dict = {'ENSG1': 1, 'ENSG2': 2}

        if expected_exception:
            with pytest.raises(expected_exception):
                sum_ensembl_ids(adata, collapse_gene_ids, gene_mapping_dict, gene_token_dict, file_format=data_format)
        else:
            result = sum_ensembl_ids(adata, collapse_gene_ids, gene_mapping_dict, gene_token_dict, file_format=data_format)
            assert isinstance(result, AnnData)
            assert result.shape == expected_shape

    def test_tokenize_anndata(self, tokenizer_v1):
        # Create a test AnnData object
        adata = AnnData(X=sp.csr_matrix([[1, 2, 3], [4, 5, 6]]))
        adata.var['ensembl_id'] = ['ENSG1', 'ENSG2', 'ENSG3']
        adata.obs['n_counts'] = [6, 15]
        adata.obs['cell_type'] = ['A', 'B']

        tokenizer_v1.custom_attr_name_dict = {'cell_type': 'cell_type'}
        tokenized_cells, cell_metadata = tokenizer_v1.tokenize_anndata(adata)

        assert len(tokenized_cells) == 2
        assert isinstance(tokenized_cells[0], np.ndarray)
        assert cell_metadata == {'cell_type': ['A', 'B']}

    @pytest.mark.parametrize("tokenizer, expected_max_length", [
        ("tokenizer_v1", 2048),
        ("tokenizer_v2", 4096)
    ])
    def test_create_dataset(self, request, tokenizer, expected_max_length):
        tokenizer = request.getfixturevalue(tokenizer)
        tokenized_cells = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        cell_metadata = None

        dataset = tokenizer.create_dataset(tokenized_cells, cell_metadata)
        assert len(dataset) == 2
        assert 'input_ids' in dataset.features
        assert 'length' in dataset.features
        assert all(len(ids) <= expected_max_length for ids in dataset['input_ids'])

        if tokenizer.special_token:
            assert all(ids[0] == tokenizer.gene_token_dict['<cls>'] for ids in dataset['input_ids'])
            assert all(ids[-1] == tokenizer.gene_token_dict['<eos>'] for ids in dataset['input_ids'])

    def test_tokenize_data(self, tokenizer_v1, tmp_path):
        # Create a test AnnData object
        adata = AnnData(X=sp.csr_matrix([[1, 2, 3], [4, 5, 6]]))
        adata.var['ensembl_id'] = ['ENSG1', 'ENSG2', 'ENSG3']
        adata.obs['n_counts'] = [6, 15]
        adata.obs['cell_type'] = ['A', 'B']
        test_file = tmp_path / "test.h5ad"
        adata.write_h5ad(test_file)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        tokenizer_v1.tokenize_data(tmp_path, output_dir, "test_output", file_format="h5ad")

        assert (output_dir / "test_output.dataset").exists()

        with pytest.raises(ValueError):
            tokenizer_v1.tokenize_data(tmp_path, output_dir, "test_output", file_format="bad_format")

