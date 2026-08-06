"""GenePT's gene-identifier handling.

GenePT's embedding table is keyed on **gene symbols** (`get_text_embeddings` looks
up `self.embeddings.get(emb.upper())` over `var_names`), so Ensembl gene IDs have
to be mapped *to* symbols -- the opposite direction from the Ensembl-keyed models.

The guard used to read `if gene_names == "ensembl_id":`, copied from Geneformer
where the correct comparison is `!=`. That made `map_ensembl_ids_to_gene_symbols`
unreachable on every input where it would have been correct, and there was no
test directory for GenePT at all, so nothing caught it.

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
# Two distinct Ensembl IDs that both map to the symbol PRPF31.
PRPF31_A, PRPF31_B = "ENSG00000274144", "ENSG00000105618"


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
        # Must name the expected symbols: asserting only "no ENSG prefixes remain"
        # is satisfied just as well by *dropping* the versioned genes, which is
        # exactly what happened before the version suffix was stripped before
        # lookup. Versioned IDs are the GENCODE/CellRanger default.
        processed = model.process_data(_adata([f"{TP53}.17", f"{ACTB}.9", GAPDH]))
        assert {"TP53", "ACTB", "GAPDH"} <= set(processed.var_names)
        assert not any(str(g).startswith("ENSG") for g in processed.var_names)

    def test_ids_colliding_on_one_symbol_are_collapsed(self, model):
        # Ensembl -> symbol is many-to-one (10616 of 48698 symbol-bearing rows in
        # hsapiens_pybiomart.csv share a gene_name). Both of these map to PRPF31;
        # without collapsing, the var index is non-unique and process_data's own
        # `adata[:, genes_names]` raises InvalidIndexError.
        processed = model.process_data(_adata([PRPF31_A, PRPF31_B, GAPDH]))
        assert list(processed.var_names).count("PRPF31") == 1
        assert processed.var_names.is_unique

    @pytest.mark.parametrize("silent_copy", [0, 1], ids=["first", "second"])
    def test_collision_keeps_the_expressed_copy(self, model, silent_copy):
        """Of a colliding set, the copy carrying the counts must win.

        Parametrised over *which* copy is silent, because that is the whole
        point: picking positionally passes for one arrangement and fails for the
        other, letting an all-zero alt-scaffold copy displace the expressed gene
        and report a real gene as unexpressed. Asserted as "the surviving column
        is non-zero" rather than against raw counts, since process_data applies
        normalize_total/log1p afterwards -- but log1p(0) is still 0, so a silent
        survivor would register.
        """
        adata = _adata([PRPF31_A, PRPF31_B, GAPDH])
        dense = adata.X.toarray()
        dense[:, silent_copy] = 0
        adata.X = sp.csr_matrix(dense)

        processed = model.process_data(adata)
        column = list(processed.var_names).index("PRPF31")
        kept = processed[:, column].X
        kept = kept.toarray() if sp.issparse(kept) else kept
        assert kept.sum() > 0, "the silent duplicate displaced the expressed copy"

    def test_ids_without_a_symbol_are_dropped(self, model):
        processed = model.process_data(_adata([TP53, ACTB, GAPDH, NO_SYMBOL]))
        assert NO_SYMBOL not in set(processed.var_names)

    def test_raises_when_nothing_maps(self, model):
        # Filler is unmappable Ensembl IDs too, so genuinely nothing resolves.
        with pytest.raises(ValueError, match="could be mapped to gene symbols"):
            model.process_data(_adata([UNKNOWN], filler="ENSG9999999{i:04d}"))

    def test_gene_names_column_is_honoured(self, model):
        # Must assert the symbols appear: the fixture's own index holds no ENSG
        # values, so "no ENSG prefixes remain" holds whether or not `my_ids` was
        # ever read -- gating the mapping on `gene_names == "index"` passed it.
        adata = _adata(["g1", "g2", "g3"])
        adata.var["my_ids"] = [TP53, ACTB, GAPDH] + list(adata.var_names[3:])
        processed = model.process_data(adata, gene_names="my_ids")
        assert {"TP53", "ACTB", "GAPDH"} <= set(processed.var_names)
