import logging
import re
from typing import List, Optional, Sequence
from helical.utils.downloader import Downloader
import numpy as np
import pandas as pd
from anndata import AnnData
from pathlib import Path
from helical.constants.paths import CACHE_DIR_HELICAL

LOGGER = logging.getLogger(__name__)


def _get_ensembl_mart_df(species: str = "hsapiens") -> pd.DataFrame:
    """
    Fetch a (species)_gene_ensembl table via pybiomart.

    Parameters
    ----------
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "ensembl_id" and "gene_name".
    """
    import pybiomart
    server = pybiomart.Server(host="http://www.ensembl.org")
    dataset = server.marts["ENSEMBL_MART_ENSEMBL"].datasets[f"{species}_gene_ensembl"]
    df = dataset.query(attributes=["ensembl_gene_id", "external_gene_name"])
    df = df.rename(columns={"Gene stable ID": "ensembl_id", "Gene name": "gene_name"})
    return df.sort_values(by="ensembl_id")


def map_gene_symbols_to_ensembl_ids(
    adata: AnnData, gene_names: Optional[str] = None, species: str = "hsapiens"
) -> AnnData:
    """
    Map gene symbols to Ensembl Gene IDs using pybiomart.

    Due to duplication events, some symbols map to multiple Ensembl IDs; we take
    the first occurrence after de-duplication.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene metadata in `adata.var`.
    gene_names : str, optional
        Column in `adata.var` containing gene symbols. If None, uses `adata.var_names`.
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    AnnData
        Same object with `adata.var["ensembl_id"]` populated.
    """
    var_names = adata.var[gene_names] if gene_names is not None else pd.Series(adata.var_names, index=adata.var_names)
    adata.var["ensembl_id"] = convert_list_gene_symbols_to_ensembl_ids(var_names, species=species)
    non_none_mappings = adata.var["ensembl_id"].notnull().sum()
    LOGGER.info("Mapped %d / %d genes to Ensembl IDs.", non_none_mappings, adata.var.shape[0])
    return adata


def map_ensembl_ids_to_gene_symbols(
    adata: AnnData, ensembl_id_key: str = "ensembl_id", species: str = "hsapiens"
) -> AnnData:
    """
    Map Ensembl Gene IDs to gene symbols using pybiomart.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing gene metadata in `adata.var`.
    ensembl_id_key : str, default "ensembl_id"
        Column in `adata.var` containing Ensembl Gene IDs.
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart (e.g., "hsapiens", "mmusculus").

    Returns
    -------
    AnnData
        Same object with `adata.var["gene_names"]` populated.
    """
    adata.var["gene_names"] = convert_list_ensembl_ids_to_gene_symbols(adata.var[ensembl_id_key], species=species)
    non_none_mappings = adata.var["gene_names"].notnull().sum()
    LOGGER.info("Mapped %d / %d Ensembl IDs to gene names.", non_none_mappings, adata.var.shape[0])
    return adata

def _load_static_ensembl_df() -> pd.DataFrame:
    """
    Load a static mapping table between gene names and ensembl ids for 'hsapiens'.
    This avoids having to call an unstable API endpoint from pybiomart.
    Instead load a static csv from helical.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns "ensembl_id" and "gene_name".
    """
    downloader = Downloader()
    downloader.download_via_name('hsapiens_pybiomart.csv')
    df_path = Path(CACHE_DIR_HELICAL, "hsapiens_pybiomart.csv")
    df = pd.read_csv(df_path, index_col=0)
    return df


def convert_list_ensembl_ids_to_gene_symbols(
    ensembl_ids: Sequence[str], species: str = "hsapiens"
) -> List[Optional[str]]:
    """
    Map a list/sequence of Ensembl Gene IDs to gene symbols using pybiomart.

    Parameters
    ----------
    ensembl_ids : Sequence[str]
        Ensembl Gene IDs (e.g., "ENSG00000139618").
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart.

    Returns
    -------
    List[Optional[str]]
        Gene symbols aligned to the input order (None if not found).
    """
    if species == "hsapiens":
        df = _load_static_ensembl_df()
    else:
        df = _get_ensembl_mart_df(species=species)
    mapping = df.drop_duplicates(subset="ensembl_id").set_index("ensembl_id")["gene_name"]
    return list(pd.Series(list(ensembl_ids), dtype="object").map(mapping).where(pd.notna, None))


def convert_list_gene_symbols_to_ensembl_ids(
    gene_symbols: Sequence[str], species: str = "hsapiens"
) -> List[Optional[str]]:
    """
    Map a list/sequence of gene symbols to Ensembl Gene IDs using pybiomart.

    Parameters
    ----------
    gene_symbols : Sequence[str]
        Gene symbols (e.g., "BRCA2", "KANSL2").
    species : str, default "hsapiens"
        Species prefix used by Ensembl Biomart.

    Returns
    -------
    List[Optional[str]]
        Ensembl Gene IDs aligned to the input order (None if not found).
    """
    if species == "hsapiens":
        df = _load_static_ensembl_df()
    else:
        df = _get_ensembl_mart_df(species=species)
    mapping = df.drop_duplicates(subset="gene_name").set_index("gene_name")["ensembl_id"]
    return list(pd.Series(list(gene_symbols), dtype="object").map(mapping).where(pd.notna, None))

# ──────────────────────────────────────────────────────────────────────────────
# Gene-identifier system: detection and normalisation
#
# Models key their vocabularies on one of two systems -- gene symbols (scGPT, UCE,
# GenePT, C2S) or Ensembl gene IDs (Geneformer, Tahoe, Nicheformer,
# Transcriptformer) -- and a dataset in the other system overlaps by exactly zero
# genes. Each model's process_data reconciles this itself; the primitives live here
# so the same subtle bug is not written five times (helicalAI/bio-agent#1117, #1123).
# ──────────────────────────────────────────────────────────────────────────────

#: Ensembl **gene** IDs, optionally version-suffixed (ENSG00000141510.17).
#: Deliberately narrower than `startswith("ENS")`, which also matches real gene
#: symbols (ENSA) and transcript/protein IDs (ENST.., ENSP..).
ENSEMBL_GENE_ID_PATTERN = r"^ENS[A-Z]{0,4}G\d{11}(\.\d+)?$"
_ENSEMBL_GENE_ID_RE = re.compile(ENSEMBL_GENE_ID_PATTERN)

#: Values that mean "an upstream mapping already failed here", not a gene name.
#: Matched exactly rather than by prefix, so a real gene whose symbol merely starts
#: with these letters (NANOS1, NAT1) is unaffected.
_NULL_IDENTIFIERS = frozenset({"none", "nan", "na", "null", ""})


def is_ensembl_gene_id(value: object) -> bool:
    """Is ``value`` an Ensembl gene ID (version suffix allowed)?"""
    return bool(_ENSEMBL_GENE_ID_RE.match(str(value)))


def ensembl_id_mask(values: Sequence[object]) -> "pd.Series":
    """Per-entry mask of which ``values`` are Ensembl gene IDs.

    Per entry rather than `.all()`/`.any()` over the column: a var index that is
    only *mostly* Ensembl IDs must convert the Ensembl entries and leave real
    symbols alone, which neither aggregate can express.
    """
    return pd.Series([is_ensembl_gene_id(v) for v in values], dtype=bool)


def strip_ensembl_version(value: object) -> str:
    """``ENSG00000141510.17`` -> ``ENSG00000141510``; anything else untouched.

    Applied only to Ensembl-matching values: real gene symbols contain dots too
    (``AC000068.10``), and truncating those would silently corrupt them. The
    mapping tables are keyed on bare IDs, so skipping this step makes every
    versioned ID -- the GENCODE/CellRanger default -- resolve to None and be
    dropped as "unmapped".
    """
    text = str(value)
    return text.split(".", 1)[0] if is_ensembl_gene_id(text) else text


def reject_null_identifiers(identifiers: Sequence[object]) -> None:
    """Raise if any identifier is a null sentinel rather than a gene name.

    A literal ``"None"`` in a gene column means an earlier mapping step already
    failed and wrote its failure into the data. Converting around it would silently
    drop those genes, hiding the original problem, so refuse the input instead.
    """
    n_null = sum(
        1 for value in identifiers if str(value).strip().lower() in _NULL_IDENTIFIERS
    )
    if n_null:
        message = (
            f"{n_null} gene identifier(s) are null placeholders ('None'/'nan'/empty) "
            f"rather than gene names, so an earlier mapping step has already failed. "
            f"Remove or repair those entries before processing the data."
        )
        LOGGER.error(message)
        raise ValueError(message)


def require_vocabulary_overlap(identifiers: Sequence[object], vocabulary) -> None:
    """Raise when no identifier is in ``vocabulary``.

    Catches the case the old namespace guards caught by accident: identifiers that
    are structurally valid but from the wrong annotation entirely -- mouse Ensembl
    IDs against a human vocabulary, say. Checking membership rather than *shape*
    means genuinely usable Ensembl input is accepted while unusable input still
    fails loudly, instead of silently tokenizing to nothing.
    """
    if not any(value and value in vocabulary for value in identifiers):
        message = (
            f"None of the gene identifiers are in the model's vocabulary "
            f"({len(vocabulary)} entries). They are well-formed, so this usually "
            f"means they are from a different annotation or species than the model "
            f"was trained on. Check .var of the anndata input object."
        )
        LOGGER.error(message)
        raise ValueError(message)


def _identifiers_of(adata: AnnData, gene_names: str) -> "pd.Series":
    """The identifier column named by ``gene_names``, as strings."""
    if gene_names == "index":
        return pd.Series(list(adata.var_names), dtype=object).astype(str)
    return adata.var[gene_names].astype(str).reset_index(drop=True)


def _collapse_duplicates(resolved: List[Optional[str]], adata: AnnData) -> "pd.Series":
    """Boolean keep-mask: drop unmapped entries and collapse duplicate names.

    Symbol collisions are real and common -- 10616 of the 48698 symbol-bearing rows
    in the bundled mapping table share a gene_name (3369 symbols carried by >=2
    ids). A non-unique var index breaks downstream indexing outright, so one column
    per name has to win.

    Of a colliding set, keep the column carrying the most counts. Choosing
    positionally lets an all-zero alt-scaffold copy displace the expressed one,
    after which the gene reads as unexpressed with no error anywhere. ``X`` is only
    touched when a name is actually claimed twice, since a backed AnnData exposes
    it as a dataset with no ``.sum`` and the common case must not pay for totals it
    would never consult.
    """
    counts: dict = {}
    for name in resolved:
        if name:
            counts[name] = counts.get(name, 0) + 1

    totals = None
    if any(count > 1 for count in counts.values()):
        matrix = adata.to_memory().X if adata.isbacked else adata.X
        totals = np.asarray(matrix.sum(axis=0)).ravel()

    best: dict = {}
    for index, name in enumerate(resolved):
        if not name:
            continue
        incumbent = best.get(name)
        if incumbent is None or (
            totals is not None and totals[index] > totals[incumbent]
        ):
            best[name] = index

    winners = set(best.values())
    return pd.Series(
        [bool(name) and index in winners for index, name in enumerate(resolved)],
        dtype=bool,
    )


def _log_accounting(resolved: List[Optional[str]], keep: "pd.Series") -> None:
    n_in = len(resolved)
    n_out = int(keep.sum())
    n_unmapped = sum(1 for name in resolved if not name)
    LOGGER.info(
        "Gene identifiers: %d in -> %d out (%d dropped with no match, "
        "%d dropped as duplicate names).",
        n_in,
        n_out,
        n_unmapped,
        n_in - n_out - n_unmapped,
    )


def ensure_gene_symbols(
    adata: AnnData,
    gene_names: str = "index",
    species: str = "hsapiens",
) -> AnnData:
    """Return ``adata`` with ``var_names`` guaranteed to be gene symbols.

    For the symbol-keyed models (scGPT, UCE, GenePT, C2S), whose vocabularies are
    looked up by symbol. Ensembl gene IDs are translated per entry; identifiers
    that are already symbols are left untouched, so a mixed index keeps both.
    Genes with no symbol, and duplicates after translation, are dropped with the
    counts logged. Returns the input unchanged when nothing is an Ensembl ID.
    """
    identifiers = _identifiers_of(adata, gene_names)
    reject_null_identifiers(identifiers)
    is_ensembl = ensembl_id_mask(identifiers)
    if not is_ensembl.any():
        return adata

    bare = sorted({strip_ensembl_version(v) for v in identifiers[is_ensembl]})
    symbols = convert_list_ensembl_ids_to_gene_symbols(bare, species=species)
    lookup = {
        ensembl_id: symbol
        for ensembl_id, symbol in zip(bare, symbols)
        if symbol  # excludes None *and* blank, so no entry resolves via a bogus key
    }
    resolved = [
        lookup.get(strip_ensembl_version(value)) if flag else value
        for value, flag in zip(identifiers, is_ensembl)
    ]

    keep = _collapse_duplicates(resolved, adata)
    _log_accounting(resolved, keep)
    if not keep.any():
        message = (
            f"None of the Ensembl gene IDs could be mapped to gene symbols, which "
            f"this model's vocabulary is keyed on. Check the identifiers in .var of "
            f"the anndata input object, and that species={species!r} matches the data."
        )
        LOGGER.error(message)
        raise ValueError(message)

    out = adata[:, keep.to_numpy()]
    out = out.to_memory() if adata.isbacked else out.copy()
    kept_names = [name for name, kept in zip(resolved, keep) if kept]
    out.var["original_gene_id"] = [
        value for value, kept in zip(identifiers, keep) if kept
    ]
    out.var_names = kept_names
    # Keep the caller's chosen identifier column in step with var_names. Callers
    # run `ensure_rna_data_validity` first, which materialises var["index"] from
    # the *pre-conversion* index, so a lookup that reads that column instead of
    # var_names would otherwise still see the old identifiers and match nothing.
    if gene_names in out.var.columns:
        out.var[gene_names] = kept_names
    return out


def ensure_ensembl_ids(
    adata: AnnData,
    gene_names: str = "index",
    species: str = "hsapiens",
) -> AnnData:
    """Return ``adata`` with a ``var["ensembl_id"]`` column, from either system.

    For the Ensembl-keyed models (Geneformer, Tahoe, Nicheformer,
    Transcriptformer). Identifiers that are already Ensembl gene IDs are used
    **directly** -- crucially *not* round-tripped through symbols, which would
    drop every gene that has no symbol (~44% of the mapping table's rows) and can
    land others on a different in-vocabulary ID entirely. Gene symbols are
    translated; unmapped entries get an empty string, never None, so they simply
    fail the vocabulary lookup instead of resolving through a bogus key.

    ``var_names`` are left alone: the caller's identifiers stay addressable, which
    is what ``id_to_gene``-style reverse lookups and caller-supplied gene lists
    need (helicalAI/bio-agent#1128).
    """
    identifiers = _identifiers_of(adata, gene_names)
    reject_null_identifiers(identifiers)
    is_ensembl = ensembl_id_mask(identifiers)

    to_convert = sorted({v for v, flag in zip(identifiers, is_ensembl) if not flag})
    lookup = {}
    if to_convert:
        mapped = convert_list_gene_symbols_to_ensembl_ids(to_convert, species=species)
        lookup = {
            symbol: ensembl_id
            for symbol, ensembl_id in zip(to_convert, mapped)
            if ensembl_id
        }

    resolved = [
        strip_ensembl_version(value) if flag else lookup.get(value, "")
        for value, flag in zip(identifiers, is_ensembl)
    ]
    if not any(resolved):
        message = (
            f"None of the gene identifiers could be resolved to Ensembl gene IDs, "
            f"which this model's vocabulary is keyed on. Check the identifiers in "
            f".var of the anndata input object, and that species={species!r} matches."
        )
        LOGGER.error(message)
        raise ValueError(message)

    LOGGER.info(
        "Gene identifiers: %d already Ensembl IDs, %d mapped from symbols, "
        "%d unresolved.",
        int(is_ensembl.sum()),
        int((~is_ensembl).sum()) - resolved.count(""),
        resolved.count(""),
    )
    out = adata if not adata.isbacked else adata.to_memory()
    out.var["ensembl_id"] = resolved
    return out
