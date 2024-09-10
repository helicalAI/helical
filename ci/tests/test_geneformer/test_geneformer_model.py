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
    data.obs["cell_type"] = ["CD4 T cells"]
    data.X = [[1, 2, 5]]
    tokenized_dataset = geneformer.process_data(data, gene_names='gene_symbols')

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

    def test_ensure_data_validity_raising_error_with_missing_ensembl_id_column(self):
        del self.data.var['ensembl_id']
        with pytest.raises(KeyError):
            self.geneformer.ensure_data_validity(self.data, "ensembl_id")
    
    @pytest.mark.parametrize("gene_symbols, raises_error",
                             [
                                (['ENSGSAMD11', 'ENSGPLEKHN1', 'ENSGHES4'], True),
                                (['SAMD11', 'None', 'HES4'], True),
                                (['SAMD11', 'PLEKHN1', 'HES4'], False),
                             ]
    )
    def test_ensembl_data_is_caught(self, gene_symbols, raises_error):
        self.data.var['gene_symbols'] = gene_symbols
        if raises_error:
            with pytest.raises(ValueError):
                self.geneformer.process_data(self.data, "gene_symbols")

    # test fine_tune_classifier method in for Geneformer class
    def test_fine_tune_classifier(self):
        #TODO: Implement this test
        assert self.geneformer.fine_tune_classifier is not None
            