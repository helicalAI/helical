from helical.models.scgpt.model import scGPT
from anndata import AnnData
from helical.models.scgpt.tokenizer import GeneVocab

class TestSCGPTModel:
    scgpt = scGPT()

    # Create a dummy AnnData object
    data = AnnData()
    data.var["gene_names"] = ['SAMD11', 'PLEKHN1', "NOT_IN_VOCAB", "<pad>", 'HES4']
    data.obs["n_counts"] = [1]
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
        dataset = self.scgpt.process_data(self.data, gene_column_name = "gene_names")

        assert self.scgpt.gene_column_name == "gene_names"
        assert self.scgpt.gene_column_name in self.data.var

        # asserts that all the genes in gene_column_name have been correctly translated 
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
        dataset = self.scgpt.process_data(self.data, gene_column_name = "gene_names", use_batch_labels=True)
        assert (dataset.batch_ids == batch_id_array).all()

    def test_direct_assignment_of_genes_to_index(self):
        self.data.var.index = ['SAMD11', 'PLEKHN1', "NOT_IN_VOCAB", "<pad>", 'HES4']
        self.scgpt.process_data(self.data, gene_column_name = "index", use_batch_labels=True)
        
        # as set above, the gene column can also be direclty assigned to the index column
        assert self.scgpt.gene_column_name == "index"
        assert self.scgpt.gene_column_name in self.data.var


    def test_get_embeddings(self):
        dataset = self.scgpt.process_data(self.data, gene_column_name = "gene_names")
        embeddings = self.scgpt.get_embeddings(dataset)
        assert embeddings.shape == (1, 512)