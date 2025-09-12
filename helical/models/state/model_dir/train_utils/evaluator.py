import logging
import multiprocessing as mp
import os
from typing import Any, Literal

import anndata as ad
import pandas as pd
import polars as pl
import scanpy as sc
from pdex import parallel_differential_expression
import numpy as np

from ._pipeline import MetricPipeline
from ._types import PerturbationAnndataPair, initialize_de_comparison

LOGGER = logging.getLogger(__name__)


class MetricsEvaluator:
    """
    Evaluates benchmarking metrics of a predicted and real anndata object.

    Arguments
    =========

    adata_pred: ad.AnnData | str
        Predicted anndata object or path to anndata object.
    adata_real: ad.AnnData | str
        Real anndata object or path to anndata object.
    de_pred: pl.DataFrame | str | None = None
        Predicted differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    de_real: pl.DataFrame | str | None = None
        Real differential expression results or path to differential expression results.
        If `None`, differential expression will be computed using parallel_differential_expression
    control_pert: str = "non-targeting"
        Control perturbation name.
    pert_col: str = "target"
        Perturbation column name.
    de_method: str = "wilcoxon"
        Differential expression method.
    num_threads: int = -1
        Number of threads for parallel differential expression.
    batch_size: int = 100
        Batch size for parallel differential expression.
    outdir: str = "./cell-eval-outdir"
        Output directory.
    allow_discrete: bool = False
        Allow discrete data.
    prefix: str | None = None
        Prefix for output files.
    pdex_kwargs: dict[str, Any] | None = None
        Keyword arguments for parallel_differential_expression.
        These will overwrite arguments passed to MetricsEvaluator.__init__ if they conflict.
    """

    def __init__(
        self,
        adata_pred: ad.AnnData | str,
        adata_real: ad.AnnData | str,
        de_pred: pl.DataFrame | str | None = None,
        de_real: pl.DataFrame | str | None = None,
        control_pert: str = "non-targeting",
        pert_col: str = "target",
        de_method: str = "wilcoxon",
        num_threads: int = -1,
        batch_size: int = 100,
        outdir: str = "./cell-eval-outdir",
        allow_discrete: bool = False,
        prefix: str | None = None,
        pdex_kwargs: dict[str, Any] | None = None,
    ):
        # Enable a global string cache for categorical columns
        pl.enable_string_cache()

        if os.path.exists(outdir):
            LOGGER.warning(
                f"Output directory {outdir} already exists, potential overwrite occurring"
            )
        os.makedirs(outdir, exist_ok=True)

        self.anndata_pair = _build_anndata_pair(
            real=adata_real,
            pred=adata_pred,
            control_pert=control_pert,
            pert_col=pert_col,
            allow_discrete=allow_discrete,
        )

        self.de_comparison = _build_de_comparison(
            anndata_pair=self.anndata_pair,
            de_pred=de_pred,
            de_real=de_real,
            de_method=de_method,
            num_threads=num_threads if num_threads != -1 else mp.cpu_count(),
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        )

        self.outdir = outdir
        self.prefix = prefix

    def compute(
        self,
        profile: Literal["full", "vcc", "minimal", "de", "anndata"] = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        skip_metrics: list[str] | None = None,
        basename: str = "results.csv",
        write_csv: bool = True,
        break_on_error: bool = False,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        pipeline = MetricPipeline(
            profile=profile,
            metric_configs=metric_configs,
            break_on_error=break_on_error,
        )
        if skip_metrics is not None:
            pipeline.skip_metrics(skip_metrics)
        pipeline.compute_de_metrics(self.de_comparison)
        pipeline.compute_anndata_metrics(self.anndata_pair)
        results = pipeline.get_results()
        agg_results = pipeline.get_agg_results()

        if write_csv:
            outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_{basename}" if self.prefix else basename,
            )
            agg_outpath = os.path.join(
                self.outdir,
                f"{self.prefix}_agg_{basename}" if self.prefix else f"agg_{basename}",
            )

            LOGGER.info(f"Writing perturbation level metrics to {outpath}")
            results.write_csv(outpath)

            LOGGER.info(f"Writing aggregate metrics to {agg_outpath}")
            agg_results.write_csv(agg_outpath)

        return results, agg_results


def _build_anndata_pair(
    real: ad.AnnData | str,
    pred: ad.AnnData | str,
    control_pert: str,
    pert_col: str,
    allow_discrete: bool = False,
    n_cells: int = 100,
):
    if isinstance(real, str):
        LOGGER.info(f"Reading real anndata from {real}")
        real = ad.read_h5ad(real)
    if isinstance(pred, str):
        LOGGER.info(f"Reading pred anndata from {pred}")
        pred = ad.read_h5ad(pred)

    # Validate that the input is normalized and log-transformed
    _convert_to_normlog(
        real, n_cells=n_cells, which="real", allow_discrete=allow_discrete
    )
    _convert_to_normlog(
        pred, n_cells=n_cells, which="pred", allow_discrete=allow_discrete
    )

    # Build the anndata pair
    return PerturbationAnndataPair(
        real=real, pred=pred, control_pert=control_pert, pert_col=pert_col
    )


def _convert_to_normlog(
    adata: ad.AnnData,
    n_cells: int | float = 5e2,
    which: str | None = None,
    allow_discrete: bool = False,
):
    """Performs a norm-log conversion if the input is integer data (inplace).

    Will skip if the input is not integer data.
    """
    if guess_is_lognorm(adata=adata, n_cells=n_cells):
        LOGGER.info(
            "Input is found to be log-normalized already - skipping transformation."
        )
        return  # Input is already log-normalized

    # User specified that they want to allow discrete data
    if allow_discrete:
        if which:
            LOGGER.info(
                f"Discovered integer data for {which}. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        else:
            LOGGER.info(
                "Discovered integer data. Configuration set to allow discrete. "
                "Make sure this is intentional."
            )
        return  # proceed without conversion

    # Convert the data to norm-log
    if which:
        LOGGER.info(f"Discovered integer data for {which}. Converting to norm-log.")
    sc.pp.normalize_total(adata=adata, inplace=True)  # normalize to median
    sc.pp.log1p(adata)  # log-transform (log1p)


def _build_de_comparison(
    anndata_pair: PerturbationAnndataPair | None = None,
    de_pred: pl.DataFrame | str | None = None,
    de_real: pl.DataFrame | str | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
):
    return initialize_de_comparison(
        real=_load_or_build_de(
            mode="real",
            de_path=de_real,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
        pred=_load_or_build_de(
            mode="pred",
            de_path=de_pred,
            anndata_pair=anndata_pair,
            de_method=de_method,
            num_threads=num_threads,
            batch_size=batch_size,
            outdir=outdir,
            prefix=prefix,
            pdex_kwargs=pdex_kwargs or {},
        ),
    )


def _build_pdex_kwargs(
    reference: str,
    groupby_key: str,
    num_workers: int,
    batch_size: int,
    metric: str,
    pdex_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pdex_kwargs = pdex_kwargs or {}
    if "reference" not in pdex_kwargs:
        pdex_kwargs["reference"] = reference
    if "groupby_key" not in pdex_kwargs:
        pdex_kwargs["groupby_key"] = groupby_key
    if "num_workers" not in pdex_kwargs:
        pdex_kwargs["num_workers"] = num_workers
    if "batch_size" not in pdex_kwargs:
        pdex_kwargs["batch_size"] = batch_size
    if "metric" not in pdex_kwargs:
        pdex_kwargs["metric"] = metric
    # always return polars DataFrames
    pdex_kwargs["as_polars"] = True
    return pdex_kwargs


def _load_or_build_de(
    mode: Literal["pred", "real"],
    de_path: pl.DataFrame | str | None = None,
    anndata_pair: PerturbationAnndataPair | None = None,
    de_method: str = "wilcoxon",
    num_threads: int = 1,
    batch_size: int = 100,
    outdir: str | None = None,
    prefix: str | None = None,
    pdex_kwargs: dict[str, Any] | None = None,
) -> pl.DataFrame:
    if de_path is None:
        if anndata_pair is None:
            raise ValueError("anndata_pair must be provided if de_path is not provided")
        LOGGER.info(f"Computing DE for {mode} data")
        pdex_kwargs = _build_pdex_kwargs(
            reference=anndata_pair.control_pert,
            groupby_key=anndata_pair.pert_col,
            num_workers=num_threads,
            metric=de_method,
            batch_size=batch_size,
            pdex_kwargs=pdex_kwargs or {},
        )
        frame = parallel_differential_expression(
            adata=anndata_pair.real if mode == "real" else anndata_pair.pred,
            **pdex_kwargs,
        )
        if outdir is not None:
            pathname = f"{mode}_de.csv" if not prefix else f"{prefix}_{mode}_de.csv"
            LOGGER.info(f"Writing {mode} DE results to: {pathname}")
            frame.write_csv(os.path.join(outdir, pathname))

        return frame  # type: ignore
    elif isinstance(de_path, str):
        LOGGER.info(f"Reading {mode} DE results from {de_path}")
        if pdex_kwargs:
            LOGGER.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.read_csv(
            de_path,
            schema_overrides={
                "target": pl.Utf8,
                "feature": pl.Utf8,
            },
        )
    elif isinstance(de_path, pl.DataFrame):
        if pdex_kwargs:
            LOGGER.warning("pdex_kwargs are ignored when reading from a CSV file")
        return de_path
    elif isinstance(de_path, pd.DataFrame):
        if pdex_kwargs:
            LOGGER.warning("pdex_kwargs are ignored when reading from a CSV file")
        return pl.from_pandas(de_path)
    else:
        raise TypeError(f"Unexpected type for de_path: {type(de_path)}")


def guess_is_lognorm(
    adata: ad.AnnData,
    n_cells: int | float = 5e2,
    epsilon: float = 1e-2,
) -> bool:
    """Guess if the input is integer counts or log-normalized.

    This is an _educated guess_ based on whether the fractional component of cell sums.
    This _will not be able_ to distinguish between normalized input and log-normalized input.

    Returns:
        bool: True if the input is lognorm, False otherwise
    """
    # Determine the number of cells to use for the guess
    n_cells = int(min(adata.shape[0], n_cells))

    # Pick a random subset of cells
    cell_mask = np.random.choice(adata.shape[0], n_cells, replace=False)

    # Sum the counts for each cell
    cell_sums = adata.X[cell_mask].sum(axis=1)  # type: ignore (can be float but super unlikely)

    # Check if any cell sum's fractional part is greater than epsilon
    return bool(np.any(np.abs((cell_sums - cell_sums.round())) > epsilon))


def split_anndata_on_celltype(
    adata: ad.AnnData,
    celltype_col: str,
) -> dict[str, ad.AnnData]:
    """Split anndata on celltype column.

    Args:
        adata: AnnData object
        celltype_col: Column name in adata.obs that contains the celltype labels

    Returns:
        dict[str, AnnData]: Dictionary of AnnData objects, keyed by celltype
    """
    if celltype_col not in adata.obs.columns:
        raise ValueError(
            f"Celltype column {celltype_col} not found in adata.obs: {adata.obs.columns}"
        )

    return {
        ct: adata[adata.obs[celltype_col] == ct]
        for ct in adata.obs[celltype_col].unique()
    }
