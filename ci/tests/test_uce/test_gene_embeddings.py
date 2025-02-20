from helical.models.uce.gene_embeddings import load_gene_embeddings_adata
from anndata import AnnData
import pandas as pd
import numpy as np
from pathlib import Path
import pytest
from pathlib import Path

CACHE_DIR_HELICAL = Path(Path.home(), ".cache", "helical", "models")


class TestUCEGeneEmbeddings:

    adata = AnnData(
        X=np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        obs=pd.DataFrame({"species": ["human", "mouse", "rat"]}),
        var=pd.DataFrame({"gene": ["gene1", "gene2", "gene3"]}),
    )
    species = ["human"]
    embedding_model = "ESM2"
    embeddings_path = Path(CACHE_DIR_HELICAL, "uce", "protein_embeddings")

    def test_load_gene_embeddings_adata_filtering_all_genes(self):
        with pytest.raises(ValueError):
            load_gene_embeddings_adata(
                self.adata, self.species, self.embedding_model, self.embeddings_path
            )

    def test_load_gene_embeddings_adata_filtering_no_genes(self):
        self.adata.var_names = ["hoxa6", "cav2", "txk"]
        anndata, mapping_dict = load_gene_embeddings_adata(
            self.adata, self.species, self.embedding_model, self.embeddings_path
        )
        assert (anndata.var_names == ["hoxa6", "cav2", "txk"]).all()
        assert (anndata.obs == self.adata.obs).all().all()
        assert (anndata.X == self.adata.X).all()
        assert len(mapping_dict["human"]) == 19790

    def test_load_gene_embeddings_adata_filtering_some_genes(self):
        self.adata.var_names = ["hoxa6", "cav2", "1"]
        anndata, mapping_dict = load_gene_embeddings_adata(
            self.adata, self.species, self.embedding_model, self.embeddings_path
        )
        assert (anndata.var_names == ["hoxa6", "cav2"]).all()
        assert (anndata.obs == self.adata.obs).all().all()
        assert (anndata.X == [[1, 2], [4, 5], [7, 8]]).all()
        assert len(mapping_dict["human"]) == 19790
