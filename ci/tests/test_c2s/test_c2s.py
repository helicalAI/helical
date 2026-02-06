import pytest
import numpy as np
import anndata as ad
from datasets import Dataset
import torch
from helical.models.c2s.model import Cell2Sen
from helical.models.c2s.config import Cell2SenConfig


@pytest.fixture(scope="module")
def sample_anndata():
    anndata = ad.read_h5ad("yolksac_human.h5ad")
    return anndata[:3, :10].copy()


@pytest.fixture(scope="module")
def base_config():
    return Cell2SenConfig(batch_size=8, max_new_tokens=10, device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="module")
def cell2sen_model(base_config):
    return Cell2Sen(configurer=base_config)


@pytest.fixture(scope="module")
def processed_dataset_basic(cell2sen_model, sample_anndata):
    return cell2sen_model.process_data(sample_anndata)


@pytest.fixture(scope="module")
def processed_dataset_with_perturbation(cell2sen_model, sample_anndata):
    data = sample_anndata.copy()
    data.obs['perturbation'] = 'IFNg'

    original = cell2sen_model.perturbation_column
    cell2sen_model.perturbation_column = 'perturbation'
    dataset = cell2sen_model.process_data(data)
    cell2sen_model.perturbation_column = original

    return dataset


@pytest.fixture(scope="module")
def processed_dataset_linear_map(cell2sen_model, sample_anndata):
    original = cell2sen_model.return_fit
    cell2sen_model.return_fit = True
    dataset = cell2sen_model.process_data(sample_anndata)
    cell2sen_model.return_fit = original
    return dataset


class TestInit:
    def test_init_default(self):
        model = Cell2Sen()
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device is not None

    def test_init_with_config(self):
        config = Cell2SenConfig(batch_size=16, organism="Homo sapiens")
        model = Cell2Sen(configurer=config)
        assert model.batch_size == 16
        assert model.organism == "Homo sapiens"

    def test_init_with_quantization(self):
        config = Cell2SenConfig(
            batch_size=8,
            model_size="2B",
            use_quantization=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = Cell2Sen(configurer=config)
        assert model.model.config.quantization_config is not None

    def test_init_with_explicit_device_cpu(self):
        config = Cell2SenConfig(device="cpu")
        model = Cell2Sen(configurer=config)
        assert model.device == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_init_with_explicit_device_cuda(self):
        config = Cell2SenConfig(device="cuda")
        model = Cell2Sen(configurer=config)
        assert model.device == "cuda"

    def test_attn_implementation_sdpa_default(self):
        config = Cell2SenConfig(device="cpu", use_flash_attn=False)
        model = Cell2Sen(configurer=config)
        assert model.attn_implementation == "sdpa"

    def test_attn_cpu_flash_attn_ignored(self):
        config = Cell2SenConfig(device="cpu", use_flash_attn=True)
        model = Cell2Sen(configurer=config)
        assert model.attn_implementation == "sdpa"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attn_flash_on_cuda(self):
        config = Cell2SenConfig(device="cuda", use_flash_attn=True)
        model = Cell2Sen(configurer=config)
        assert model.attn_implementation == "flash_attention_2"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attn_sdpa_without_flash_cuda(self):
        config = Cell2SenConfig(device="cuda", use_flash_attn=False)
        model = Cell2Sen(configurer=config)
        assert model.attn_implementation == "sdpa"

    def test_model_in_eval_mode(self, cell2sen_model):
        assert not cell2sen_model.model.training

    def test_tokenizer_loaded(self, cell2sen_model):
        out = cell2sen_model.tokenizer("Test", return_tensors="pt")
        assert 'input_ids' in out


class TestProcessData:
    def test_process_data_basic(self, cell2sen_model, sample_anndata):
        dataset = cell2sen_model.process_data(sample_anndata)
        assert isinstance(dataset, Dataset)
        assert len(dataset) == sample_anndata.n_obs
        expected = ['cell_sentence', 'fit_parameters', 'organism', 'perturbations']
        assert all(c in dataset.column_names for c in expected)
        assert all(isinstance(s, str) for s in dataset['cell_sentence'])
        assert all(org == dataset['organism'][0] for org in dataset['organism'])
        assert all(p is None for p in dataset['perturbations'])

    def test_process_data_max_genes(self, sample_anndata):
        config = Cell2SenConfig(max_genes=5)
        model = Cell2Sen(configurer=config)
        dataset = model.process_data(sample_anndata)
        for s in dataset['cell_sentence']:
            if s:
                assert len(s.split()) <= 5

    def test_process_data_max_genes_none(self, sample_anndata):
        config = Cell2SenConfig(max_genes=None)
        model = Cell2Sen(configurer=config)
        _ = model.process_data(sample_anndata)

    def test_process_with_perturbation(self, cell2sen_model, sample_anndata):
        data = sample_anndata.copy()
        data.obs['perturbation'] = 'IFNg'
        original = cell2sen_model.perturbation_column
        cell2sen_model.perturbation_column = 'perturbation'
        dataset = cell2sen_model.process_data(data)
        cell2sen_model.perturbation_column = original
        assert all(p == 'IFNg' for p in dataset['perturbations'])

    def test_process_organism(self, cell2sen_model, sample_anndata):
        original = cell2sen_model.organism
        cell2sen_model.organism = "Homo sapiens"
        dataset = cell2sen_model.process_data(sample_anndata)
        cell2sen_model.organism = original
        assert all(org == "Homo sapiens" for org in dataset['organism'])

    def test_process_fit_parameters(self, processed_dataset_linear_map):
        params = processed_dataset_linear_map['fit_parameters'][0]
        if params:
            assert 'slope' in params

    def test_process_fit_parameters_disabled(self, processed_dataset_basic):
        assert all(p is None for p in processed_dataset_basic['fit_parameters'])


class TestGetEmbeddings:
    def test_embeddings_shape(self, cell2sen_model, processed_dataset_basic):
        emb = cell2sen_model.get_embeddings(processed_dataset_basic)
        assert isinstance(emb, np.ndarray)
        assert emb.ndim == 2
        assert emb.shape[0] == len(processed_dataset_basic)
        assert emb.shape[1] == 2304

    def test_batch_size_variation(self, cell2sen_model, processed_dataset_basic):
        orig = cell2sen_model.batch_size
        cell2sen_model.batch_size = 2
        e1 = cell2sen_model.get_embeddings(processed_dataset_basic)
        cell2sen_model.batch_size = 3
        e2 = cell2sen_model.get_embeddings(processed_dataset_basic)
        cell2sen_model.batch_size = orig
        assert e1.shape == e2.shape

    def test_empty_dataset(self, cell2sen_model, sample_anndata):
        with pytest.raises(Exception):
            empty = cell2sen_model.process_data(sample_anndata[:0])
            cell2sen_model.get_embeddings(empty)

    def test_attention_shapes(self, cell2sen_model, processed_dataset_basic):
        emb, attn, gene_order = cell2sen_model.get_embeddings(processed_dataset_basic, output_attentions=True)
        assert isinstance(attn, list)
        assert isinstance(gene_order, list)
        assert len(gene_order) == len(processed_dataset_basic)
        assert all(isinstance(gene_list, list) for gene_list in gene_order)
        assert len(attn) == len(processed_dataset_basic)
        for sample_attn in attn:
            # (num_heads, num_genes, num_genes)
            assert sample_attn.ndim == 3
            assert sample_attn.shape[1] == sample_attn.shape[2]


class TestGetPerturbations:
    def test_comprehensive(self, cell2sen_model, processed_dataset_with_perturbation):
        orig = cell2sen_model.batch_size
        cell2sen_model.batch_size = 5
        dataset, perturbed = cell2sen_model.get_perturbations(processed_dataset_with_perturbation)
        cell2sen_model.batch_size = orig
        assert len(perturbed) == len(dataset)
        assert all(isinstance(p, str) for p in perturbed)

    def test_with_list(self, cell2sen_model, processed_dataset_basic):
        plist = ['IFNg'] * len(processed_dataset_basic)
        _, perturbed = cell2sen_model.get_perturbations(processed_dataset_basic, perturbations_list=plist)
        assert all(isinstance(p, str) for p in perturbed)

    def test_mixed_none(self, cell2sen_model, sample_anndata):
        data = sample_anndata.copy()
        vals = ['IFNg', None, 'TNF'] * (len(data) // 3 + 1)
        data.obs['perturbation'] = vals[: len(data)]
        original = cell2sen_model.perturbation_column
        cell2sen_model.perturbation_column = 'perturbation'
        dataset = cell2sen_model.process_data(data)
        dataset, pert = cell2sen_model.get_perturbations(dataset)
        cell2sen_model.perturbation_column = original
        assert any(p is None for p in pert)
        assert any(p is not None for p in pert)

    def test_error_all_none(self, cell2sen_model, processed_dataset_basic):
        with pytest.raises(ValueError):
            cell2sen_model.get_perturbations(processed_dataset_basic)

    def test_error_length_mismatch(self, cell2sen_model, processed_dataset_basic):
        with pytest.raises(ValueError):
            cell2sen_model.get_perturbations(processed_dataset_basic, perturbations_list=['IFNg'] * 5)
