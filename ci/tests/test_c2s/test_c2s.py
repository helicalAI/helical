"""
Unit tests for Cell2Sen model using pytest.
"""
import pytest
import numpy as np
import anndata as ad
from datasets import Dataset
import torch
from helical.models.c2s.model import Cell2Sen
from helical.models.c2s.config import Cell2SenConfig

@pytest.fixture(scope="module")
def sample_anndata():
    """Load and prepare a small sample of the yolksac dataset for testing."""
    anndata = ad.read_h5ad("yolksac_human.h5ad")
    # Use a small subset for faster testing
    anndata = anndata[:3, :10].copy()
    return anndata


@pytest.fixture(scope="module")
def cell2sen_model():
    """Initialize Cell2Sen model once for all tests."""
    config = Cell2SenConfig(batch_size=8)
    return Cell2Sen(configurer=config)


@pytest.fixture(scope="module")
def processed_dataset_basic(cell2sen_model, sample_anndata):
    """Pre-processed dataset for basic tests."""
    return cell2sen_model.process_data(sample_anndata)


@pytest.fixture(scope="module")
def processed_dataset_with_perturbation(cell2sen_model, sample_anndata):
    """Pre-processed dataset with perturbation column."""
    sample_anndata_copy = sample_anndata.copy()
    sample_anndata_copy.obs['perturbation'] = 'IFNg'
    
    # Temporarily modify the shared model's perturbation_column
    original_pert_col = cell2sen_model.perturbation_column
    cell2sen_model.perturbation_column = 'perturbation'
    
    dataset = cell2sen_model.process_data(sample_anndata_copy)
    cell2sen_model.perturbation_column = original_pert_col
    
    return dataset


@pytest.fixture(scope="module")
def processed_dataset_linear_map(cell2sen_model, sample_anndata):
    """Pre-processed dataset with linear expression map enabled."""
    # Temporarily modify the shared model's return_fit
    original_return_fit = cell2sen_model.return_fit
    cell2sen_model.return_fit = True
    
    dataset = cell2sen_model.process_data(sample_anndata)
    cell2sen_model.return_fit = original_return_fit
    
    return dataset


class TestInit:
    """Test Cell2Sen initialization."""
    
    def test_init_default(self):
        """Test model initialization with default parameters."""
        model = Cell2Sen()
        assert model is not None
        assert hasattr(model, 'model')
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'device')
        assert model.model is not None
        assert model.tokenizer is not None
    
    def test_init_with_config(self):
        """Test model initialization with custom config."""
        config = Cell2SenConfig(batch_size=16, organism="Homo sapiens")
        model = Cell2Sen(configurer=config)
        assert model.batch_size == 16
        assert model.organism == "Homo sapiens"
    
    def test_init_with_quantization(self):
        """Test model initialization with quantization."""
        config = Cell2SenConfig(batch_size=8, model_size="2B", use_quantization=True)
        model = Cell2Sen(configurer=config)
        assert model.model is not None
        assert model.tokenizer is not None
        assert model.device == "cuda" if torch.cuda.is_available() else "cpu"
        assert model.model.config.quantization_config is not None
    
    def test_model_on_correct_device(self, cell2sen_model):
        """Test that model is on the correct device."""
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"
        assert cell2sen_model.device == expected_device
    
    def test_model_in_eval_mode(self, cell2sen_model):
        """Test that model is in evaluation mode."""
        assert not cell2sen_model.model.training
    
    def test_tokenizer_loaded(self, cell2sen_model):
        """Test that tokenizer is properly loaded."""
        test_text = "Test sentence"
        tokens = cell2sen_model.tokenizer(test_text, return_tensors="pt")
        assert tokens is not None
        assert 'input_ids' in tokens


class TestProcessData:
    """Test process_data method."""
    
    def test_process_data_basic_properties(self, cell2sen_model, sample_anndata):
        """Test basic data processing - checks all properties in one go."""
        dataset = cell2sen_model.process_data(sample_anndata)
        
        # Check dataset type and structure
        assert isinstance(dataset, Dataset)
        assert len(dataset) == sample_anndata.n_obs
        
        # Check all required columns exist
        required_columns = ['cell_sentence', 'fit_parameters', 'organism', 'perturbations']
        assert all(col in dataset.column_names for col in required_columns)
        
        # Check cell sentences format
        assert all(isinstance(s, str) for s in dataset['cell_sentence'])
        assert len(dataset['cell_sentence'][0].split()) > 0
        
        # Check organism format
        assert all(isinstance(org, str) for org in dataset['organism'])
        assert all(org == dataset['organism'][0] for org in dataset['organism'])
        
        # Check perturbations are None when not provided
        assert all(p is None for p in dataset['perturbations'])
    
    def test_process_data_with_perturbation(self, cell2sen_model, sample_anndata):
        """Test data processing with perturbation column."""
        sample_anndata_copy = sample_anndata.copy()
        sample_anndata_copy.obs['perturbation'] = 'IFNg'
        
        # Temporarily modify the shared model
        original_pert_col = cell2sen_model.perturbation_column
        cell2sen_model.perturbation_column = 'perturbation'
        
        dataset = cell2sen_model.process_data(sample_anndata_copy)
        assert len(dataset['perturbations']) == len(dataset)
        assert all(p == 'IFNg' for p in dataset['perturbations'])
        cell2sen_model.perturbation_column = original_pert_col
    
    def test_process_data_with_perturbation_list(self, cell2sen_model, sample_anndata):
        """Test data processing with list of perturbations."""
        sample_anndata_copy = sample_anndata.copy()
        perturbations = ['IFNg', 'TNF', 'IL-1'] * (sample_anndata.n_obs // 3 + 1)
        perturbations = perturbations[:sample_anndata.n_obs]
        sample_anndata_copy.obs['perturbation'] = perturbations
        
        # Temporarily modify the shared model
        original_pert_col = cell2sen_model.perturbation_column
        cell2sen_model.perturbation_column = 'perturbation'
        
        dataset = cell2sen_model.process_data(sample_anndata_copy)
        assert len(dataset['perturbations']) == len(dataset)
        assert dataset['perturbations'][0] == 'IFNg'
        cell2sen_model.perturbation_column = original_pert_col
    
    def test_process_data_organism_provided(self, cell2sen_model, sample_anndata):
        """Test that organism is properly stored when provided."""
        organism = "Homo sapiens"
        
        # Temporarily modify the shared model
        original_organism = cell2sen_model.organism
        cell2sen_model.organism = organism
        
        dataset = cell2sen_model.process_data(sample_anndata)
        assert 'organism' in dataset.column_names
        assert len(dataset['organism']) == len(dataset)
        assert all(isinstance(org, str) for org in dataset['organism'])
        assert all(org == organism for org in dataset['organism'])
        cell2sen_model.organism = original_organism
    
    def test_process_data_fit_parameters(self, processed_dataset_linear_map):
        """Test that fit parameters are calculated correctly."""
        assert len(processed_dataset_linear_map['fit_parameters']) == len(processed_dataset_linear_map)
        first_params = processed_dataset_linear_map['fit_parameters'][0]
        if first_params is not None:
            assert 'slope' in first_params
            assert 'intercept' in first_params
            assert 'r_squared' in first_params
    
    def test_process_data_fit_parameters_disabled(self, processed_dataset_basic):
        """Test that fit parameters are None when disabled."""
        assert all(fp is None for fp in processed_dataset_basic['fit_parameters'])


class TestGetEmbeddings:
    """Test get_embeddings method."""
    
    def test_get_embeddings_comprehensive(self, cell2sen_model, processed_dataset_basic):
        """Test embeddings comprehensively - shape, type, dimensions."""
        embeddings = cell2sen_model.get_embeddings(processed_dataset_basic)
        
        # Check type and shape
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.ndim == 2
        assert embeddings.shape[0] == len(processed_dataset_basic)
        assert embeddings.shape[1] == 2304  # Gemma 2 2B hidden size
        assert embeddings.dtype == np.float32
        
    
    def test_get_embeddings_with_different_batch_sizes(self, cell2sen_model, processed_dataset_basic):
        """Test embeddings have consistent shape across batch sizes."""
        # Use the shared model but change batch size dynamically
        original_batch_size = cell2sen_model.batch_size
        
        # Test with batch size 2
        cell2sen_model.batch_size = 2
        embeddings_batch2 = cell2sen_model.get_embeddings(processed_dataset_basic)
        
        # Test with batch size 3
        cell2sen_model.batch_size = 3
        embeddings_batch5 = cell2sen_model.get_embeddings(processed_dataset_basic)
        
        # Only check shape - values may differ due to padding differences
        assert embeddings_batch2.shape == embeddings_batch5.shape
        assert embeddings_batch2.shape[0] == len(processed_dataset_basic)
        # Restore original batch size
        cell2sen_model.batch_size = original_batch_size

    
    def test_get_embeddings_empty_dataset(self, cell2sen_model, sample_anndata):
        """Test embeddings with empty dataset."""
        empty_dataset = cell2sen_model.process_data(sample_anndata[:0])
        with pytest.raises((ValueError, IndexError, AssertionError)):
            cell2sen_model.get_embeddings(empty_dataset)

    def test_get_embeddings_attention_shapes(self, cell2sen_model, processed_dataset_basic):
        """Test that attention maps have correct shapes when output_attentions=True."""
        embeddings, attentions = cell2sen_model.get_embeddings(
            processed_dataset_basic, 
            output_attentions=True
        )
        
        # Check embeddings shape
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(processed_dataset_basic)
        
        # Check attentions is a tuple (one element per layer)
        assert isinstance(attentions, tuple)
        assert len(attentions) > 0  # Should have at least one layer
        
        # Check each layer's attention map shape
        num_cells = len(processed_dataset_basic)
        for layer_idx, attn_layer in enumerate(attentions):
            assert isinstance(attn_layer, np.ndarray)
            assert attn_layer.ndim == 4, f"Layer {layer_idx} attention should be 4D"
            assert attn_layer.shape[0] == num_cells, f"Layer {layer_idx} batch dimension should match number of cells"
            assert attn_layer.shape[1] > 0, f"Layer {layer_idx} should have at least one attention head"
            assert attn_layer.shape[2] == attn_layer.shape[3], f"Layer {layer_idx} attention should be square (seq_len x seq_len)"
            assert attn_layer.shape[2] > 0, f"Layer {layer_idx} sequence length should be positive"

class TestGetPerturbations:
    """Test get_perturbations method."""
    
    def test_get_perturbations_comprehensive(self, cell2sen_model, processed_dataset_with_perturbation):
        """Test perturbation generation comprehensively."""
        # Temporarily modify batch_size
        original_batch_size = cell2sen_model.batch_size
        cell2sen_model.batch_size = 5
        
        dataset, perturbed_sentences = cell2sen_model.get_perturbations(processed_dataset_with_perturbation)
        assert isinstance(perturbed_sentences, list)
        assert len(perturbed_sentences) == len(dataset)
        assert all(isinstance(s, str) or s is None for s in perturbed_sentences)
        assert 'perturbed_cell_sentence' in dataset.column_names
        assert len(dataset['perturbed_cell_sentence']) == len(dataset)
        assert all(s is not None for s in perturbed_sentences)
        assert all(isinstance(s, str) for s in perturbed_sentences)
        assert all(len(s) > 0 for s in perturbed_sentences)
        cell2sen_model.batch_size = original_batch_size
    
    def test_get_perturbations_with_perturbations_list(self, cell2sen_model, processed_dataset_basic):
        """Test perturbation generation with provided perturbations_list."""
        perturbations_list = ['IFNg'] * len(processed_dataset_basic)
        dataset, perturbed_sentences = cell2sen_model.get_perturbations(
            processed_dataset_basic,
            perturbations_list=perturbations_list
        )
        
        assert len(perturbed_sentences) == len(processed_dataset_basic)
        assert all(s is not None for s in perturbed_sentences)
        
    
    def test_get_perturbations_with_mixed_none(self, cell2sen_model, sample_anndata):
        """Test perturbation generation with mixed None values."""
        sample_anndata_copy = sample_anndata.copy()
        perturbations = ['IFNg', None, 'TNF', None, 'IL-1'] * (sample_anndata.n_obs // 5 + 1)
        perturbations = perturbations[:sample_anndata.n_obs]
        sample_anndata_copy.obs['perturbation'] = perturbations
        
        # Temporarily modify the shared model
        original_pert_col = cell2sen_model.perturbation_column
        cell2sen_model.perturbation_column = 'perturbation'
        
        dataset = cell2sen_model.process_data(sample_anndata_copy)
        dataset, perturbed_sentences = cell2sen_model.get_perturbations(dataset)
        assert len(perturbed_sentences) == len(dataset)
        assert any(s is None for s in perturbed_sentences)
        assert any(s is not None for s in perturbed_sentences)
        cell2sen_model.perturbation_column = original_pert_col
    
    def test_get_perturbations_all_none_error(self, cell2sen_model, processed_dataset_basic):
        """Test error when all perturbations are None."""
        with pytest.raises(ValueError, match="No valid perturbations"):
            cell2sen_model.get_perturbations(processed_dataset_basic)
    
    def test_get_perturbations_length_mismatch_error(self, cell2sen_model, processed_dataset_basic):
        """Test error when perturbations_list length doesn't match dataset."""
        perturbations_list = ['IFNg'] * 5  # Wrong length
        
        with pytest.raises(ValueError, match="must match dataset length"):
            cell2sen_model.get_perturbations(
                processed_dataset_basic,
                perturbations_list=perturbations_list
            )