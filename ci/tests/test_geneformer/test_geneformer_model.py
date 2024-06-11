import pytest
from helical.models.geneformer.model import Geneformer
from anndata import AnnData
import numpy as np
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL

class TestGeneformerModel:
    geneformer = Geneformer()

    # Create a dummy AnnData object
    data = AnnData()
    data.var['gene_symbols'] = ['SAMD11', 'PLEKHN1', 'HES4']
    data.obs["n_counts"] = [1]
    data.obs["cell_type"] = ["CD4 T cells"]
    data.X = [[1, 2, 5]]
    tokenized_dataset = geneformer.process_data(data)

    def test_process_data_mapping_to_ensemble_ids(self):
        assert self.data.var['ensembl_id'][0] == 'ENSG00000187634'
        # is the same as the above line but more verbose (linking the gene symbol to the ensembl id)
        assert self.data.var[self.data.var['gene_symbols'] == 'SAMD11']['ensembl_id'].values[0] == 'ENSG00000187634'
        assert self.data.var[self.data.var['gene_symbols'] == 'PLEKHN1']['ensembl_id'].values[0] == 'ENSG00000187583'
        assert self.data.var[self.data.var['gene_symbols'] == 'HES4']['ensembl_id'].values[0] == 'ENSG00000188290'

    def test_process_data_padding_and_masking_ids(self):
        # for this token mapping, the padding token is 0 and the mask token is 1
        assert self.geneformer.gene_token_dict.get("<pad>") == 0
        assert self.geneformer.gene_token_dict.get("<mask>") == 1

    miss_cell_type = AnnData()
    miss_cell_type.obs["n_counts"] = [1]
    miss_cell_type.var["gene_symbols"] = [1]

    miss_n_counts = AnnData()
    miss_n_counts.obs["cell_type"] = [1]
    miss_n_counts.var["gene_symbols"] = [1]

    miss_gene_symbols = AnnData()
    miss_gene_symbols.obs["cell_type"] = [1]
    miss_gene_symbols.obs["n_counts"] = [1]

    miss_ensembl_id = AnnData()
    miss_ensembl_id.obs["cell_type"] = [1]
    miss_ensembl_id.obs["n_counts"] = [1]
    miss_ensembl_id.var["gene_symbols"] = [1]

    @pytest.mark.parametrize("data, use_gene_symbols", 
                             [
                                (miss_cell_type, False),
                                (miss_n_counts, False),
                                (miss_gene_symbols, True),
                                (miss_ensembl_id, False)
                             ]
    )
    def test_check_data_eligibility(self, data, use_gene_symbols):
        with pytest.raises(KeyError):
            self.geneformer._check_data_eligibility(data, use_gene_symbols)

class TestTranscriptomeTokenizer:
    model_dir = Path(CACHE_DIR_HELICAL, 'geneformer')
    files_config = {
        "gene_median_path": model_dir / "gene_median_dictionary.pkl",
        "token_path": model_dir / "token_dictionary.pkl"
    }

    tokenizer = TranscriptomeTokenizer({"cell_type": "cell_type"},
                                        nproc = 4, 
                                        gene_median_file = files_config["gene_median_path"], 
                                        token_dictionary_file = files_config["token_path"])
    
    

    @pytest.mark.parametrize("number_of_obs, x_data_count, expected_token", 
                             [
                                #  number of observations has no effect on the resulting, expected tokens
                                (3, [5, 2, 5], [16026, 16175, 16012]),
                                (5, [5, 2, 5], [16026, 16175, 16012]),
                                #  x_data_count has an effect on the expected tokens
                                (5, [7, 88, 43], [16012, 16175, 16026]),
                                (5, [780, 1, 1], [16026, 16012, 16175]),
                             ]
    )
    def test_tokenize_anndata(self, number_of_obs: int, x_data_count: list[int], expected_token: list[int]):
        """
        Test the `tokenize_anndata` method of the tokenizer.
        The x_data_count and cell_type will be the same for each observation in the test data.
        What is being tested is the tokenization of the x_data_count and the corresponding ensemble_id.  
        What can be seen in this test is that the same tokens are expected for the same three ensemble_ids.
        The order in which these tokens are listed, depends among other things on the x_data_count.

        Args:
            number_of_obs (int): The number of observations in the test data.
            x_data_count: The count of x data.
            expected_token: The expected token to compare against.
        """

        data = AnnData()
        data.var['ensembl_id'] = ['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290']
        data.obs["n_counts"] = [1] * number_of_obs
        data.obs["cell_type"] = ["CD4 T cells"] * number_of_obs
        data.X = [x_data_count] * number_of_obs

        tokenized_cells, cell_metadata =  self.tokenizer.tokenize_anndata(data)
        for tokenized_cell in tokenized_cells:
            assert np.array_equal(tokenized_cell, expected_token)
        assert len(tokenized_cells) == number_of_obs
        assert cell_metadata == {'cell_type': ['CD4 T cells'] * number_of_obs}
        
        # test the created dataset containing essentially the same information as the tokenized_cells variable
        tokenized_dataset = self.tokenizer.create_dataset(tokenized_cells, cell_metadata)
        assert tokenized_dataset.shape == (number_of_obs, 3)
        assert tokenized_dataset['cell_type'] == ['CD4 T cells'] * number_of_obs
        assert tokenized_dataset['length'] == [3] * number_of_obs
        for tokenized_cell_dataset in tokenized_dataset['input_ids']:
            assert np.array_equal(tokenized_cell_dataset, expected_token)
