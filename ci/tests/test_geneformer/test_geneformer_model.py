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

    miss_n_counts = AnnData()
    miss_n_counts.var["gene_symbols"] = [1]

    miss_gene_symbols = AnnData()
    miss_gene_symbols.obs["n_counts"] = [1]

    miss_ensembl_id = AnnData()
    miss_ensembl_id.obs["n_counts"] = [1]
    miss_ensembl_id.var["gene_symbols"] = [1]

    @pytest.mark.parametrize("data, use_gene_symbols", 
                             [
                                (miss_n_counts, True),
                                (miss_gene_symbols, True),
                                (miss_ensembl_id, False)
                             ]
    )
    def test_ensure_data_validity(self, data, use_gene_symbols):
        with pytest.raises(KeyError):
            self.geneformer.ensure_data_validity(data, use_gene_symbols)
            