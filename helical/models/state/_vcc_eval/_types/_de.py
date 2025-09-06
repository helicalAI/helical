import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Iterator, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from ._enums import DESortBy

logger = logging.getLogger(__name__)


def initialize_de_comparison(
    real: pl.DataFrame,
    pred: pl.DataFrame,
    target_col: str = "target",
    feature_col: str = "feature",
    fold_change_col: str = "fold_change",
    log2_fold_change_col: str = "log2_fold_change",
    abs_log2_fold_change_col: str = "abs_log2_fold_change",
    pvalue_col: str = "p_value",
    fdr_col: str = "fdr",
) -> "DEComparison":
    partial_de_result = partial(
        DEResults,
        target_col=target_col,
        feature_col=feature_col,
        fold_change_col=fold_change_col,
        log2_fold_change_col=log2_fold_change_col,
        abs_log2_fold_change_col=abs_log2_fold_change_col,
    )
    with pl.StringCache():
        return DEComparison(real=partial_de_result(real), pred=partial_de_result(pred))


@dataclass(frozen=False)
class DEResults:
    """Raw differential expression results with sorting and filtering capabilities."""

    data: pl.DataFrame

    # Column names configuration
    target_col: str = "target"
    feature_col: str = "feature"
    fold_change_col: str = "fold_change"
    log2_fold_change_col: str = "log2_fold_change"
    abs_log2_fold_change_col: str = "abs_log2_fold_change"
    pvalue_col: str = "p_value"
    fdr_col: str = "fdr"

    def __post_init__(self) -> None:
        required_cols = {
            self.target_col,
            self.feature_col,
            self.fold_change_col,
            self.pvalue_col,
            self.fdr_col,
        }
        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        numeric_cols = [
            self.fold_change_col,
            self.pvalue_col,
            self.fdr_col,
            self.log2_fold_change_col,
            self.abs_log2_fold_change_col,
        ]

        categorical_cols = [
            self.target_col,
            self.feature_col,
        ]

        # Add log2 fold change columns if not present
        if self.log2_fold_change_col not in self.data.columns:
            self.data = self.data.with_columns(
                pl.col(self.fold_change_col)
                .log(base=2)
                .alias(self.log2_fold_change_col)
                .fill_nan(0.0)
            ).with_columns(
                pl.col(self.log2_fold_change_col)
                .abs()
                .alias(self.abs_log2_fold_change_col)
            )

        # Enforce types
        self.data = self.data.with_columns(
            [pl.col(c).cast(pl.Float32) for c in numeric_cols]
            + [pl.col(c).cast(pl.Categorical) for c in categorical_cols]
        ).drop(
            [c for c in self.data.columns if c not in numeric_cols + categorical_cols]
        )

    def get_perts(self) -> NDArray[np.str_]:
        """Get perturbations."""
        perts = self.data[self.target_col].unique().to_numpy()
        perts.sort()
        return perts

    def get_significant_genes(
        self, pert: str, fdr_threshold: float = 0.05
    ) -> NDArray[np.str_]:
        """Get significant genes for a perturbation."""
        return (
            self.data.filter(
                (pl.col(self.target_col) == pert)
                & (pl.col(self.fdr_col) < fdr_threshold)
            )
            .select(self.feature_col)
            .to_numpy()
        )

    def filter_to_significant(
        self,
        fdr_threshold: float = 0.05,
    ) -> pl.DataFrame:
        """Filter DE results to significant genes."""
        return self.data.filter(pl.col(self.fdr_col) < fdr_threshold)

    def get_top_genes(
        self,
        sort_by: DESortBy,
        fdr_threshold: float | None = None,
    ) -> pl.DataFrame:
        """Get top genes per perturbation, optionally filtered by FDR."""
        # Set FDR threshold if not provided
        fdr_threshold = fdr_threshold if fdr_threshold is not None else 0.05

        descending = sort_by in {DESortBy.FOLD_CHANGE, DESortBy.ABS_FOLD_CHANGE}

        # Create a rank matrix where each row is the ordinal rank of a gene and each column is a perturbation.
        # The rank is sensitive to the sort-by column and is computed post-filtering for FDR.
        rank_matrix = (
            self.data.filter(pl.col(self.fdr_col) < fdr_threshold)
            .with_columns(
                rank=pl.struct(sort_by.value)
                .rank("ordinal", descending=descending)
                .over("target")
                - 1
            )
            .pivot(
                index="rank",
                on="target",
                values="feature",
            )
            .sort("rank")
        )

        # Add perturbations that are missing from the rank matrix (no significant genes)
        missing_perts = set(self.get_perts()) - set(rank_matrix.columns)
        if missing_perts:
            rank_matrix = rank_matrix.with_columns(
                [pl.lit(None).alias(p) for p in missing_perts]
            )

        return rank_matrix


@dataclass
class DEComparison:
    """Comparison between real and predicted DE results."""

    real: DEResults
    pred: DEResults

    perturbations: NDArray[np.str_] = field(init=False)
    n_perts: int = field(init=False)

    def __post_init__(self) -> None:
        real_perts = self.real.get_perts()
        pred_perts = self.pred.get_perts()
        if not np.array_equal(real_perts, pred_perts):
            raise ValueError(
                f"Perturbation mismatch: real {real_perts} != pred {pred_perts}"
            )

        object.__setattr__(self, "perturbations", list(real_perts))
        object.__setattr__(self, "n_perts", len(real_perts))

    def iter_perturbations(self) -> Iterator[str]:
        for pert in self.perturbations:
            yield pert

    def get_perts(self, include_control: bool = False) -> NDArray[np.str_]:
        """Get perturbations."""
        if include_control:
            logger.warning("DEComparison should not include control perturbation")
        return self.perturbations

    def compute_overlap(
        self,
        k: int | None,
        metric: Literal["overlap", "precision"] = "overlap",
        fdr_threshold: float | None = None,
        sort_by: DESortBy = DESortBy.ABS_FOLD_CHANGE,
    ) -> dict[str, float]:
        """
        Compute overlap metrics across perturbations.

        Args:
            k: If specified, use top k genes from real data
            fdr_threshold: If specified, only consider genes below this FDR
            sort_by: Metric to sort genes by

        Returns:
            Dictionary mapping perturbation names to overlap scores
        """
        real_sig_rank_matrix = self.real.get_top_genes(
            sort_by=sort_by, fdr_threshold=fdr_threshold
        )
        pred_sig_rank_matrix = self.pred.get_top_genes(
            sort_by=sort_by, fdr_threshold=fdr_threshold
        )

        if real_sig_rank_matrix.shape[0] == 0 or pred_sig_rank_matrix.shape[0] == 0:
            # No significant genes in either real or predicted DE.
            # Cannot evaluate in this case so setting all perturbations to 0.0
            return {pert: 0.0 for pert in self.iter_perturbations()}

        overlaps = {}
        for pert in self.iter_perturbations():
            # If perturbation is not in either real or pred, set overlap to 0.0
            if (
                pert not in real_sig_rank_matrix.columns
                or pert not in pred_sig_rank_matrix.columns
            ):
                overlaps[pert] = 0.0
                continue

            # Get sorted gene lists
            real_genes = real_sig_rank_matrix[pert].drop_nulls().to_numpy()
            pred_genes = pred_sig_rank_matrix[pert].drop_nulls().to_numpy()

            match metric:
                case "overlap":
                    k_eff = real_genes.size if not k else k
                    k_eff = min(k_eff, real_genes.size)
                case "precision":
                    k_eff = pred_genes.size if not k else k
                    k_eff = min(k_eff, pred_genes.size)
                case _:
                    raise ValueError(f"Invalid metric: {metric}")

            if k_eff == 0:
                overlaps[pert] = 0.0
            else:
                real_subset = real_genes[:k_eff]
                pred_subset = pred_genes[:k_eff]
                overlaps[pert] = np.intersect1d(real_subset, pred_subset).size / k_eff

        return overlaps
