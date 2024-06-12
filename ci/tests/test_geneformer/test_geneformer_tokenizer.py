import pytest
from helical.models.geneformer.geneformer_tokenizer import TranscriptomeTokenizer
from anndata import AnnData
from helical.constants.paths import CACHE_DIR_HELICAL
from pathlib import Path
import numpy as np

class TestTranscriptomeTokenizer:
    model_dir = Path(CACHE_DIR_HELICAL, 'geneformer')
    files_config = {
        "gene_median_path": model_dir / "gene_median_dictionary.pkl",
        "token_path": model_dir / "token_dictionary.pkl"
    }
    tokenizer = TranscriptomeTokenizer(custom_attr_name_dict = {"cell_type": "cell_type"},
                                        nproc = 4, 
                                        gene_median_file = files_config["gene_median_path"], 
                                        token_dictionary_file = files_config["token_path"])

    data_w_filter_pass = AnnData(np.array([[5], [5], [5], [5], [5], [5]]))
    data_w_filter_pass.var['gene_symbols'] = ['a']
    data_w_filter_pass.obs["filter_pass"] = [1, 0, 1, 1, 0, 1]

    data_without_filter_pass = AnnData(np.array([[5], [5], [5], [5], [5], [5]]))
    data_without_filter_pass.var['gene_symbols'] = ['a']

    @pytest.mark.parametrize("data, expected_result", 
                             [
                                #  the idea is to only tokenize where the filter is 1 
                                (data_w_filter_pass, [0, 2, 3, 5]),
                                #  no 'filter_pass' in the obs of the anndata object, thus tokenize all the genes
                                (data_without_filter_pass, [0, 1, 2, 3, 4, 5]),
                             ]
    )
    def test_get_filter_pass_loc(self, data, expected_result):
        """
        Test that the _get_filter_pass_loc method of the GeneFormerTokenizer class correctly retrieves the indices of the 'filter_pass'
        column in the AnnData object where the value is 1. The idea is to only tokenize where the filter is 1. 
        If there is no 'filter_pass' in the obs of the anndata object, the function should return all the indices, thus tokenize all the genes.
        """
        # Call the _get_filter_pass_loc method
        filter_pass_loc = self.tokenizer._get_filter_pass_loc(data)

        # Check the result
        assert all(filter_pass_loc == expected_result)


    @pytest.mark.parametrize("ensembl_ids, x_data_count, expected_token", 
                             [
                                #  find the corresponding tokens for each ensemble id
                                (['ENSG00000187634'], [2], [16026]),
                                (['ENSG00000187583'], [2], [16012]),
                                (['ENSG00000188290'], [2], [16175]),
                                # x_data_count has an effect on the expected tokens: they are ranked in descending order of x_data_count
                                (['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290'], [11, 99, 55], [16012, 16175, 16026]),
                                (['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290'], [99, 11, 55], [16026, 16175, 16012]),
                                # TODO figure out how the count influences the tokenization as shown with the following tests:
                                (['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290'], [999, 55, 11], [16026, 16012, 16175]),
                                (['ENSG00000187634', 'ENSG00000187583', 'ENSG00000188290'], [99, 55, 11], [16012, 16026, 16175]),

                             ]
    )
    def test_tokenize_anndata(self, ensembl_ids: list[str], x_data_count: list[int], expected_token: list[int]):
        """
        Test the `tokenize_anndata` method of the tokenizer.
        The ensemble ids are correclty mapped to the right token. And the resulting tokenized_cells have the right order of this list of tokens.

        Args:
            ensembl_id (list[str]): The ensembl ids.
            x_data_count: The count of x data.
            expected_token: The expected token to compare against.
        """
        number_of_obs = 5
        data = AnnData()
        data.var['ensembl_id'] = ensembl_ids
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
        assert tokenized_dataset['length'] == [len(ensembl_ids)] * number_of_obs
        for tokenized_cell_dataset in tokenized_dataset['input_ids']:
            assert np.array_equal(tokenized_cell_dataset, expected_token)
