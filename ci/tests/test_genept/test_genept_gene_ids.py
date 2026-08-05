"""GenePT's gene-identifier handling.

GenePT's embedding table is keyed on **gene symbols** (`get_text_embeddings` looks
up `self.embeddings.get(emb.upper())` over `var_names`), so Ensembl gene IDs have
to be mapped *to* symbols -- the opposite direction from the Ensembl-keyed models.

The guard used to read `if gene_names == "ensembl_id":`, copied from Geneformer
where the correct comparison is `!=`. That made `map_ensembl_ids_to_gene_symbols`
unreachable on every input where it would have been correct, and there was no
test directory for GenePT at all, so nothing caught it (bio-agent#1117).

`GenePT.__new__` is used to skip `__init__`, which would download the embedding
table; `process_data` needs no instance state beyond the inherited validity check.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from helical.models.genept import GenePT

# Real human genes, so the static hsapiens_pybiomart.csv table resolves them.
TP53, ACTB, GAPDH = "ENSG00000141510", "ENSG00000075624", "ENSG00000111640"
# In the mapping table, but with a blank symbol.
NO_SYMBOL = "ENSG00000159239"
UNKNOWN = "ENSG99999999999"


@pytest.fixture
def model():
    return GenePT.__new__(GenePT)


# process_data runs highly_variable_genes(flavor="seurat_v3"), whose loess fit
# segfaults on a handful of genes. Pad every fixture out to a workable size with
# symbol-shaped filler, which the mapping leaves untouched. n_top_genes defaults
# above this count, so every gene stays "highly variable" and the assertions below
# are about the mapping, not about HVG selection.
_PAD_TO = 60


def _adata(gene_ids, n_cells=20, filler="FILLER{i}"):
    padded = list(gene_ids) + [
        filler.format(i=i) for i in range(max(0, _PAD_TO - len(gene_ids)))
    ]
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 50, size=(n_cells, len(padded))).astype(np.float32)
    adata = AnnData(X=sp.csr_matrix(counts))
    adata.var_names = padded
    return adata


class TestGenePTGeneIdentifiers:
    def test_ensembl_ids_are_mapped_to_symbols(self, model):
        processed = model.process_data(_adata([TP53, ACTB, GAPDH]))
        assert {"TP53", "ACTB", "GAPDH"} <= set(processed.var_names)
        assert not any(str(g).startswith("ENSG") for g in processed.var_names)

    def test_symbols_are_left_untouched(self, model):
        processed = model.process_data(_adata(["TP53", "ACTB", "GAPDH"]))
        assert {"TP53", "ACTB", "GAPDH"} <= set(processed.var_names)

    def test_mixed_index_keeps_its_symbols(self, model):
        # Regression guard for a `.startswith("ENS").any()` + wholesale-remap fix:
        # that would map the symbols to NaN and drop them.
        processed = model.process_data(_adata([TP53, "ACTB", GAPDH]))
        assert "ACTB" in set(processed.var_names)

    def test_symbol_beginning_with_ens_is_not_treated_as_an_ensembl_id(self, model):
        # ENSA is a real gene symbol; an unanchored `startswith("ENS")` check maps
        # it to NaN and drops it.
        processed = model.process_data(_adata(["ENSA", "TP53", "ACTB", "GAPDH"]))
        assert "ENSA" in set(processed.var_names)

    def test_version_suffixed_ids_are_mapped(self, model):
        processed = model.process_data(_adata([f"{TP53}.17", f"{ACTB}.9", GAPDH]))
        assert not any(str(g).startswith("ENSG") for g in processed.var_names)

    def test_ids_without_a_symbol_are_dropped(self, model):
        processed = model.process_data(_adata([TP53, ACTB, GAPDH, NO_SYMBOL]))
        assert NO_SYMBOL not in set(processed.var_names)

    def test_raises_when_nothing_maps(self, model):
        # Filler is unmappable Ensembl IDs too, so genuinely nothing resolves.
        with pytest.raises(ValueError, match="None of the Ensembl IDs"):
            model.process_data(_adata([UNKNOWN], filler="ENSG9999999{i:04d}"))

    def test_gene_names_column_is_honoured(self, model):
        adata = _adata(["g1", "g2", "g3"])
        adata.var["my_ids"] = [TP53, ACTB, GAPDH] + list(adata.var_names[3:])
        processed = model.process_data(adata, gene_names="my_ids")
        assert not any(str(g).startswith("ENSG") for g in processed.var_names)
