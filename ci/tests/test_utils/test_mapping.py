from helical.utils.mapping import map_gene_symbols_to_ensembl_ids
from helical.utils.mapping import map_ensembl_ids_to_gene_symbols
from helical.utils.mapping import convert_list_ensembl_ids_to_gene_symbols, convert_list_gene_symbols_to_ensembl_ids
import anndata as ad
import pytest
import requests
from unittest import mock
import helical.utils.mapping as mapping_module

adata = ad.read_h5ad("ci/tests/data/cell_type_sample.h5ad")


def test_map_gene_symbols_to_ensembl_ids():
    """
    Test if the gene symbols are mapped to Ensembl IDs correctly.
    CD99 should be mapped to ENSG00000002586.
    """
    adata.var["gene_names"] = ["CD99"] * adata.var.shape[0]
    map_gene_symbols_to_ensembl_ids(adata, gene_names="gene_names", species='hsapiens')
    assert all(adata.var["ensembl_id"] == ["ENSG00000002586"] * adata.var.shape[0])


def test_map_ensembl_ids_to_gene_symbols():
    """
    Test if the Ensembl IDs are mapped to gene symbols correctly.
    ENSG00000002330 should be mapped to BAD.
    """
    adata.var["ensembl_id"] = ["ENSG00000002330"] * adata.var.shape[0]
    map_ensembl_ids_to_gene_symbols(adata, ensembl_id_key="ensembl_id", species='hsapiens')
    assert all(adata.var["gene_names"] == ["BAD"] * adata.var.shape[0])


@pytest.mark.skip(
    reason="This test may take a long time to run because the dataset of macaque needs to be downloaded."
)
def test_map_gene_symbols_to_ensembl_ids_macaque():
    """
    Test if the gene symbols are mapped to Ensembl IDs correctly for macaque.
    CD99 should be mapped to ENSMFAG00000000608.
    Note, this test may be long the first time it is being run because the database for macaque needs to be downloaded.
    """
    adata.var["gene_names"] = ["CD99"] * adata.var.shape[0]
    map_gene_symbols_to_ensembl_ids(adata, gene_names="gene_names", species='mfascicularis')
    assert all(adata.var["ensembl_id"] == ["ENSMFAG00000000608"] * adata.var.shape[0])


def test_convert_list_ensembl_ids_to_gene_symbols():
    ensembl_ids = ["ENSG00000139618", "ENSG00000139620"]
    gene_symbols = convert_list_ensembl_ids_to_gene_symbols(ensembl_ids, species='hsapiens')
    assert gene_symbols == ["BRCA2", "KANSL2"]


def test_convert_list_gene_symbols_to_ensembl_ids():
    gene_symbols = ["BRCA2", "KANSL2"]
    ensembl_ids = convert_list_gene_symbols_to_ensembl_ids(gene_symbols, species='hsapiens')
    assert ensembl_ids == ["ENSG00000139618", "ENSG00000139620"]


TSV = "Gene stable ID\tGene name\nENSMUSG2\tB\nENSMUSG1\tA\n[success]\n"
OUTAGE_PAGE = "<html><head><title>Service unavailable</title></head></html>"


def _response(text, status_code=200):
    return mock.Mock(text=text, status_code=status_code)


@pytest.fixture
def biomart():
    """Patch the HTTP call and the backoff sleep, with an empty table cache."""
    mapping_module._get_ensembl_mart_df.cache_clear()
    with (
        mock.patch.object(mapping_module.requests, "get") as get,
        mock.patch.object(mapping_module.time, "sleep") as sleep,
    ):
        yield get, sleep
    mapping_module._get_ensembl_mart_df.cache_clear()


def test_biomart_table_is_fetched_once_over_https(biomart):
    get, _ = biomart
    get.return_value = _response(TSV)

    mapping_module.convert_list_gene_symbols_to_ensembl_ids(["A"], species="mmusculus")
    symbols = mapping_module.convert_list_ensembl_ids_to_gene_symbols(["ENSMUSG2"], species="mmusculus")

    assert symbols == ["B"]
    assert get.call_count == 1
    assert get.call_args.args[0].startswith("https://")


def test_biomart_retries_with_exponential_backoff(biomart):
    get, sleep = biomart
    get.side_effect = [_response(OUTAGE_PAGE), requests.ConnectionError("reset"), _response(TSV)]

    assert len(mapping_module._get_ensembl_mart_df("mmusculus")) == 2
    assert [c.args[0] for c in sleep.call_args_list] == [2, 4]


def test_biomart_gives_up_after_the_last_attempt(biomart):
    get, _ = biomart
    get.return_value = _response(OUTAGE_PAGE)

    with pytest.raises(RuntimeError, match="unavailable after 5 attempts"):
        mapping_module._get_ensembl_mart_df("mmusculus")
    assert get.call_count == 5


def test_biomart_unknown_species_is_not_retried(biomart):
    get, sleep = biomart
    get.return_value = _response("Query ERROR: caught BioMart::Exception::Usage: Dataset x_gene_ensembl NOT FOUND")

    with pytest.raises(ValueError, match="NOT FOUND"):
        mapping_module._get_ensembl_mart_df("x")
    sleep.assert_not_called()


def test_biomart_refuses_a_species_that_would_change_the_query(biomart):
    get, _ = biomart

    with pytest.raises(ValueError, match="Invalid species"):
        mapping_module._get_ensembl_mart_df('x"/><Attribute name="y')
    get.assert_not_called()
