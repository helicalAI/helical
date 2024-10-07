from helical.models.scgpt.model import scGPT, scGPTConfig
from helical.models.scgpt.fine_tuning_model import scGPTFineTuningModel
from anndata import AnnData
from helical.models.scgpt.tokenizer import GeneVocab
import pytest
import anndata as ad
import numpy as np
from scipy.sparse import csr_matrix

class TestSCGPTModel:
    scgpt = scGPT()

    # Create a dummy AnnData object
    data = AnnData()
    data.var["gene_names"] = ['SAMD11', 'PLEKHN1', "NOT_IN_VOCAB", "<pad>", 'HES4']
    data.obs["cell_type"] = ["CD4 T cells"]
    
    vocab = {
        "SAMD11": 0,
        "PLEKHN1": 1,
        "HES4": 2,
        "<pad>": 3,
    }
    scgpt.vocab = GeneVocab.from_dict(vocab)
    
    data.X = [[1, 2, 5, 6, 0]]
    
    def test_process_data(self):
        dataset = self.scgpt.process_data(self.data, gene_names = "gene_names")

        assert self.scgpt.gene_names == "gene_names"
        assert self.scgpt.gene_names in self.data.var

        # asserts that all the genes in gene_names have been correctly translated 
        # to the corresponding ids based on the vocabulary
        assert (self.data.var["id_in_vocab"] == [0, 1, -1, 3, 2]).all()

        # make sure that the genes not present in the vocabulary are filtered out
        # meaning -1 is not present in the gene_ids
        assert (dataset.gene_ids == [0, 1, 3, 2]).all()

        # ensure that the default index of the vocabulary is set to the id of the pad token
        assert self.scgpt.vocab.get_default_index() == 3

        assert (dataset.count_matrix == [1, 2, 6, 0]).all()

    def test_correct_handling_of_batch_ids(self):
        batch_id_array = [1]
        self.data.obs["batch_id"] = batch_id_array
        dataset = self.scgpt.process_data(self.data, gene_names = "gene_names", use_batch_labels=True)
        assert (dataset.batch_ids == batch_id_array).all()

    def test_direct_assignment_of_genes_to_index(self):
        self.data.var.index = ['SAMD11', 'PLEKHN1', "NOT_IN_VOCAB", "<pad>", 'HES4']
        self.scgpt.process_data(self.data, gene_names = "index")
        
        # as set above, the gene column can also be direclty assigned to the index column
        assert self.scgpt.gene_names == "index"
        assert self.scgpt.gene_names in self.data.var


    def test_get_embeddings(self):
        dataset = self.scgpt.process_data(self.data, gene_names = "gene_names")
        embeddings = self.scgpt.get_embeddings(dataset)
        assert embeddings.shape == (1, 512)

    dummy_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    @pytest.mark.parametrize("data, gene_names, batch_labels", 
                             [
                                #  missing gene_names in data.var
                                (AnnData(), "gene_names", False),
                                #  missing batch_id in data.obs
                                (dummy_data, "index", True),
                             ]
    )
    def test_ensure_data_validity__key_error(self, data, gene_names, batch_labels):
        with pytest.raises(KeyError):
            self.scgpt.ensure_data_validity(data, gene_names, batch_labels)
    
    err_np_arr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    err_np_arr_data.X.dtype=float
    err_np_arr_data.X[0,0] = 0.5

    err_csr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    err_csr_data.X = csr_matrix(np.random.rand(100, 5), dtype=np.float32)
    @pytest.mark.parametrize("data",
                             [
                                (err_np_arr_data),
                                (err_csr_data),
                             ]
    )
    def test_ensure_data_validity__value_error(self, data):
        '''The data in X must be ints. Test an error is raised for both np.ndarray and csr_matrix.'''
        with pytest.raises(ValueError):
            self.scgpt.ensure_data_validity(data, "index", False)
        assert "total_counts" in data.obs

    np_arr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    csr_data = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")
    csr_data.X = csr_matrix(np.random.poisson(1, size=(100, 5)), dtype=np.float32)
    @pytest.mark.parametrize("data",
                             [
                                (np_arr_data),
                                (csr_data),
                             ]
    )
    def test_ensure_data_validity__no_error(self, data):
        '''The data in X must be ints. Test no error is raised for both np.ndarray and csr_matrix.'''
        self.scgpt.ensure_data_validity(data, "index", False)
        assert "total_counts" in data.obs

    def test_fine_tune_classification_returns_correct_shape(self):
        labels = list([0])
        fine_tuned_model = scGPTFineTuningModel(scGPTConfig(), fine_tuning_head="classification", output_size=1)
        tokenized_dataset = fine_tuned_model.process_data(self.data)
        fine_tuned_model.train(train_input_data=tokenized_dataset, train_labels=labels)
        assert fine_tuned_model is not None
        outputs = fine_tuned_model.get_outputs(tokenized_dataset)
        assert outputs.shape == (len(self.data), len(labels))