import pytest
import numpy as np
import torch
import anndata as ad
from anndata import AnnData
from scipy.sparse import csr_matrix
from datasets import Dataset

from helical.models.nicheformer import Nicheformer, NicheformerConfig


@pytest.fixture
def _mocks(mocker):
    """Patch all I/O so Nicheformer can be instantiated without network or disk access."""
    mocker.patch("helical.models.nicheformer.model.Downloader")

    mock_tokenizer = mocker.MagicMock()

    def _tokenize(adata, **kwargs):
        n = adata.n_obs
        return {
            "input_ids": torch.zeros((n, 1500), dtype=torch.long),
            "attention_mask": torch.ones((n, 1500), dtype=torch.bool),
        }

    mock_tokenizer.side_effect = _tokenize
    mocker.patch(
        "helical.models.nicheformer.model.AutoTokenizer.from_pretrained",
        return_value=mock_tokenizer,
    )

    mock_model = mocker.MagicMock()

    def _get_embeddings(input_ids, attention_mask, layer, with_context):
        return torch.zeros((input_ids.shape[0], 512))

    mock_model.get_embeddings.side_effect = _get_embeddings
    mock_model.to.return_value = mock_model
    mocker.patch(
        "helical.models.nicheformer.model.AutoModelForMaskedLM.from_pretrained",
        return_value=mock_model,
    )

    return mock_tokenizer, mock_model


@pytest.fixture
def nicheformer(_mocks):
    return Nicheformer()


@pytest.fixture
def mock_adata():
    adata = AnnData(X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32))
    adata.obs_names = ["cell1", "cell2", "cell3"]
    adata.var_names = ["GENE1", "GENE2", "GENE3"]
    return adata


@pytest.fixture
def mock_adata_with_obs(mock_adata):
    adata = mock_adata.copy()
    adata.obs["modality"] = ["dissociated", "spatial", "dissociated"]
    adata.obs["specie"] = ["human", "human", "mouse"]
    adata.obs["assay"] = ["10x 3' v3", "MERFISH", "10x 3' v2"]
    return adata


class TestNicheformerProcessData:
    def test_returns_dataset(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert isinstance(dataset, Dataset)

    def test_dataset_has_input_ids_column(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert "input_ids" in dataset.features

    def test_dataset_has_attention_mask_column(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert "attention_mask" in dataset.features

    def test_dataset_length_matches_n_obs(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert len(dataset) == mock_adata.n_obs

    def test_input_ids_sequence_length(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert len(dataset["input_ids"][0]) == 1500

    def test_attention_mask_is_boolean(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        assert np.array(dataset["attention_mask"]).dtype == bool

    def test_obs_metadata_columns_accepted(self, nicheformer, mock_adata_with_obs):
        dataset = nicheformer.process_data(mock_adata_with_obs)
        assert len(dataset) == mock_adata_with_obs.n_obs

    def test_sparse_matrix_input_accepted(self, nicheformer, mock_adata):
        mock_adata.X = csr_matrix(mock_adata.X)
        dataset = nicheformer.process_data(mock_adata)
        assert len(dataset) == mock_adata.n_obs

    def test_float_counts_raises_value_error(self, nicheformer):
        adata = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
        adata.X = adata.X.astype(float)
        adata.X[0, 0] = 0.5
        with pytest.raises(ValueError):
            nicheformer.process_data(adata, gene_names="index")

    def test_missing_gene_names_column_raises_key_error(self, nicheformer, mock_adata):
        with pytest.raises(KeyError):
            nicheformer.process_data(mock_adata, gene_names="nonexistent_col")


class TestNicheformerGetEmbeddings:
    def test_returns_ndarray(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        embeddings = nicheformer.get_embeddings(dataset)
        assert isinstance(embeddings, np.ndarray)

    def test_embedding_shape(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)
        embeddings = nicheformer.get_embeddings(dataset)
        assert embeddings.shape == (mock_adata.n_obs, 512)

    def test_batching_produces_same_shape(self, nicheformer, mock_adata):
        dataset = nicheformer.process_data(mock_adata)

        nicheformer.config["batch_size"] = 1
        embeddings_bs1 = nicheformer.get_embeddings(dataset)

        nicheformer.config["batch_size"] = 32
        embeddings_bs32 = nicheformer.get_embeddings(dataset)

        assert embeddings_bs1.shape == embeddings_bs32.shape

    def test_layer_forwarded_to_model(self, nicheformer, mock_adata, _mocks):
        _, mock_model = _mocks
        nicheformer.config["layer"] = 6
        dataset = nicheformer.process_data(mock_adata)
        nicheformer.get_embeddings(dataset)
        assert mock_model.get_embeddings.call_args.kwargs["layer"] == 6

    def test_with_context_forwarded_to_model(self, nicheformer, mock_adata, _mocks):
        _, mock_model = _mocks
        nicheformer.config["with_context"] = True
        dataset = nicheformer.process_data(mock_adata)
        nicheformer.get_embeddings(dataset)
        assert mock_model.get_embeddings.call_args.kwargs["with_context"] is True


class TestNicheformerTechnologyMean:
    def test_none_does_not_call_load(self, _mocks, mocker):
        mock_tokenizer, _ = _mocks
        Nicheformer(NicheformerConfig(technology_mean=None))
        mock_tokenizer._load_technology_mean.assert_not_called()

    def test_ndarray_calls_load_with_array(self, _mocks):
        mock_tokenizer, _ = _mocks
        arr = np.ones(100)
        Nicheformer(NicheformerConfig(technology_mean=arr))
        mock_tokenizer._load_technology_mean.assert_called_once_with(arr)

    def test_path_string_calls_load_with_path(self, _mocks):
        mock_tokenizer, _ = _mocks
        Nicheformer(NicheformerConfig(technology_mean="path/to/mean.npy"))
        mock_tokenizer._load_technology_mean.assert_called_once_with("path/to/mean.npy")
