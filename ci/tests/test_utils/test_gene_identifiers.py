"""Gene-identifier detection and normalisation primitives (helicalAI/bio-agent#1123).

These replace five inlined copies of `startswith("ENS")` across the models. All
three of the following were live bugs in that expression, so each has a test here:

- it matches real gene symbols (`ENSA`) and transcript/protein IDs (`ENST…`, `ENSP…`);
- `.all()`/`.any()` over the column cannot express a var index that is only
  *mostly* Ensembl IDs;
- version suffixes are never stripped, so `ENSG…​.17` — the GENCODE/CellRanger
  default — fails a lookup keyed on bare IDs and the gene is dropped.

Uses the bundled static hsapiens table, so no network access.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from anndata import AnnData

from helical.utils.mapping import (
    ensembl_id_mask,
    ensure_ensembl_ids,
    ensure_gene_symbols,
    is_ensembl_gene_id,
    reject_null_identifiers,
    require_vocabulary_overlap,
    strip_ensembl_version,
)

TP53, ACTB, GAPDH = "ENSG00000141510", "ENSG00000075624", "ENSG00000111640"
# In Geneformer's vocabulary, but with a blank gene symbol in the mapping table.
NO_SYMBOL = "ENSG00000159239"
# Two distinct Ensembl IDs that both resolve to the symbol PRPF31.
PRPF31_A, PRPF31_B = "ENSG00000274144", "ENSG00000105618"
MOUSE = "ENSMUSG00000021033"


def _adata(identifiers, n_cells=4):
    counts = np.random.default_rng(0).integers(1, 50, size=(n_cells, len(identifiers)))
    adata = AnnData(X=sp.csr_matrix(counts.astype(np.float32)))
    adata.var_names = list(identifiers)
    return adata


class TestDetection:
    @pytest.mark.parametrize("value", [TP53, f"{TP53}.17", MOUSE])
    def test_accepts_gene_ids(self, value):
        assert is_ensembl_gene_id(value)

    @pytest.mark.parametrize(
        "value",
        [
            "ENSA",  # a real gene symbol beginning with ENS
            "ENST00000141510",  # transcript, not gene
            "ENSP00000141510",  # protein, not gene
            "TP53",
            "AC000068.10",  # a symbol containing a dot
            "ENSG123",  # too few digits
        ],
    )
    def test_rejects_non_gene_ids(self, value):
        assert not is_ensembl_gene_id(value)

    def test_mask_is_per_entry_not_aggregate(self):
        # The case neither .all() nor .any() can express.
        mask = ensembl_id_mask(["TP53", TP53, "ENSA", f"{ACTB}.3"])
        assert list(mask) == [False, True, False, True]

    @pytest.mark.parametrize("ratio_ensembl", [0.1, 0.5, 0.9])
    def test_mixed_indexes_at_several_ratios(self, ratio_ensembl):
        n = 20
        n_ensembl = int(n * ratio_ensembl)
        values = [TP53] * n_ensembl + ["TP53"] * (n - n_ensembl)
        assert ensembl_id_mask(values).sum() == n_ensembl

    def test_strip_version_only_touches_gene_ids(self):
        assert strip_ensembl_version(f"{TP53}.17") == TP53
        assert strip_ensembl_version(TP53) == TP53
        # A real symbol containing a dot must survive intact.
        assert strip_ensembl_version("AC000068.10") == "AC000068.10"


class TestEnsureGeneSymbols:
    def test_translates_ensembl_to_symbols(self):
        out = ensure_gene_symbols(_adata([TP53, ACTB, GAPDH]))
        assert list(out.var_names) == ["TP53", "ACTB", "GAPDH"]
        assert list(out.var["original_gene_id"]) == [TP53, ACTB, GAPDH]

    def test_is_a_noop_for_symbols(self):
        adata = _adata(["TP53", "ACTB"])
        assert ensure_gene_symbols(adata) is adata

    def test_keeps_symbols_in_a_mixed_index(self):
        out = ensure_gene_symbols(_adata([TP53, "ACTB", "ENSA"]))
        assert set(out.var_names) == {"TP53", "ACTB", "ENSA"}

    def test_strips_versions_before_lookup(self):
        out = ensure_gene_symbols(_adata([f"{TP53}.17", f"{ACTB}.9"]))
        assert list(out.var_names) == ["TP53", "ACTB"]

    def test_drops_ids_with_no_symbol(self):
        out = ensure_gene_symbols(_adata([TP53, NO_SYMBOL]))
        assert list(out.var_names) == ["TP53"]

    def test_collapses_colliding_symbols(self):
        out = ensure_gene_symbols(_adata([PRPF31_A, PRPF31_B, GAPDH]))
        assert out.var_names.is_unique
        assert list(out.var_names).count("PRPF31") == 1

    @pytest.mark.parametrize("silent", [0, 1], ids=["first", "second"])
    def test_collision_keeps_the_expressed_copy(self, silent):
        # Parametrised over which copy is silent: positional selection passes one
        # arrangement and fails the other, so one fixture would prove nothing.
        adata = _adata([PRPF31_A, PRPF31_B, GAPDH])
        dense = adata.X.toarray()
        dense[:, silent] = 0
        expected = dense[:, 1 - silent].sum()
        adata.X = sp.csr_matrix(dense)

        out = ensure_gene_symbols(adata)
        column = list(out.var_names).index("PRPF31")
        assert out.X.toarray()[:, column].sum() == expected

    def test_keeps_the_named_column_in_step_with_var_names(self):
        # Callers run ensure_rna_data_validity first, which materialises
        # var["index"] from the pre-conversion index; a lookup reading that column
        # would otherwise still see the old identifiers.
        adata = _adata([TP53, ACTB])
        adata.var["index"] = adata.var_names
        out = ensure_gene_symbols(adata, "index")
        assert list(out.var["index"]) == list(out.var_names) == ["TP53", "ACTB"]

    def test_raises_when_nothing_maps(self):
        with pytest.raises(ValueError, match="could be mapped to gene symbols"):
            ensure_gene_symbols(_adata(["ENSG99999999999", "ENSG99999999998"]))


class TestEnsureEnsemblIds:
    def test_uses_existing_ensembl_ids_directly(self):
        # The point of the design: no symbol round trip, so a gene with no symbol
        # at all still reaches an Ensembl-keyed vocabulary.
        out = ensure_ensembl_ids(_adata([TP53, NO_SYMBOL]))
        assert list(out.var["ensembl_id"]) == [TP53, NO_SYMBOL]

    def test_translates_symbols(self):
        out = ensure_ensembl_ids(_adata(["TP53", "ACTB"]))
        assert list(out.var["ensembl_id"]) == [TP53, ACTB]

    def test_handles_a_mixed_index(self):
        out = ensure_ensembl_ids(_adata([TP53, "ACTB"]))
        assert list(out.var["ensembl_id"]) == [TP53, ACTB]

    def test_strips_versions(self):
        out = ensure_ensembl_ids(_adata([f"{TP53}.17"]))
        assert list(out.var["ensembl_id"]) == [TP53]

    def test_unmapped_symbols_become_empty_not_none(self):
        # A None reaching a dict key or an `in` check is the aliasing trap from
        # bio-agent#1112; an empty string simply fails the vocabulary lookup.
        out = ensure_ensembl_ids(_adata(["TP53", "NOT_A_REAL_GENE_XYZ"]))
        assert list(out.var["ensembl_id"]) == [TP53, ""]

    def test_var_names_are_left_alone(self):
        # Reverse lookups (id_to_gene) and caller-supplied gene lists address the
        # caller's own identifiers, so these must not be rewritten.
        out = ensure_ensembl_ids(_adata(["TP53", "ACTB"]))
        assert list(out.var_names) == ["TP53", "ACTB"]

    def test_raises_when_nothing_resolves(self):
        with pytest.raises(ValueError, match="could be resolved to Ensembl"):
            ensure_ensembl_ids(_adata(["NOT_A_GENE_1", "NOT_A_GENE_2"]))


class TestGuards:
    @pytest.mark.parametrize("sentinel", ["None", "nan", "", "NA", "null"])
    def test_null_identifiers_are_rejected(self, sentinel):
        with pytest.raises(ValueError, match="null placeholders"):
            reject_null_identifiers(["TP53", sentinel])

    def test_real_symbols_starting_with_sentinel_letters_are_fine(self):
        # Matched exactly, not by prefix: NANOS1 and NAT1 are real genes.
        reject_null_identifiers(["NANOS1", "NAT1", "NONO"])

    def test_vocabulary_overlap_rejects_another_species(self):
        # What the removed namespace guards caught by accident: well-formed IDs
        # from the wrong annotation would otherwise tokenize to nothing, silently.
        with pytest.raises(ValueError, match="vocabulary"):
            require_vocabulary_overlap([MOUSE], {TP53: 1, ACTB: 2})

    def test_vocabulary_overlap_accepts_a_partial_match(self):
        require_vocabulary_overlap([MOUSE, TP53], {TP53: 1})

    def test_vocabulary_overlap_ignores_unresolved_entries(self):
        with pytest.raises(ValueError, match="vocabulary"):
            require_vocabulary_overlap(["", ""], {TP53: 1})
