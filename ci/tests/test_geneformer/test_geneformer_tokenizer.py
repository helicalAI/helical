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
    tokenizer = TranscriptomeTokenizer(gene_median_file=files_config["gene_median_path"], token_dictionary_file=files_config["token_path"])

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