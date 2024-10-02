from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from helical.utils.mapping import map_ensembl_ids_to_gene_symbols
from pyensembl.species import human
from pyensembl.species import macaque
import anndata as ad
import pytest

adata = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")

def test_map_gene_symbols_to_ensembl_ids():
    """
    Test if the gene symbols are mapped to Ensembl IDs correctly.
    CD99 should be mapped to ENSG00000002586.
    """
    adata.var["gene_names"] = ["CD99"] * adata.var.shape[0]
    map_gene_symbols_to_ensembl_ids(adata, gene_names = "gene_names", species = human)
    assert all(adata.var["ensembl_id"] == ["ENSG00000002586"] * adata.var.shape[0])

def test_map_ensembl_ids_to_gene_symbols():
    """
    Test if the Ensembl IDs are mapped to gene symbols correctly.
    ENSG00000002330 should be mapped to BAD.
    """
    adata.var["ensembl_id"] = ["ENSG00000002330"] * adata.var.shape[0]
    map_ensembl_ids_to_gene_symbols(adata, ensembl_id_key="ensembl_id", species = human)
    assert all(adata.var["gene_names"] == ["BAD"] * adata.var.shape[0])

@pytest.mark.skip(reason="This test may take a long time to run because the dataset of macaque needs to be downloaded.")
def test_map_gene_symbols_to_ensembl_ids_macaque():
    """
    Test if the gene symbols are mapped to Ensembl IDs correctly for macaque.
    CD99 should be mapped to ENSMFAG00000000608.
    Note, this test may be long the first time it is being run because the database for macaque needs to be downloaded.
    """
    adata.var["gene_names"] = ["CD99"] * adata.var.shape[0]
    map_gene_symbols_to_ensembl_ids(adata, gene_names = "gene_names", species = macaque)
    assert all(adata.var["ensembl_id"] == ["ENSMFAG00000000608"] * adata.var.shape[0])