"""Metrics package for evaluating cell perturbation predictions."""

from ._anndata import (
    ClusteringAgreement,
    discrimination_score,
    mae,
    mae_delta,
    mse,
    mse_delta,
    pearson_delta,
)
from ._de import (
    DEDirectionMatch,
    DENsigCounts,
    DESigGenesRecall,
    DESpearmanLFC,
    DESpearmanSignificant,
    compute_pr_auc,
    compute_roc_auc,
    de_overlap_metric,
)
from ._impl import metrics_registry
from .base import Metric, MetricInfo, MetricResult

__all__ = [
    # Array metrics
    "ClusteringAgreement",
    "pearson_delta",
    "mse",
    "mae",
    "mse_delta",
    "mae_delta",
    "discrimination_score",
    # DE metrics
    "DEDirectionMatch",
    "DESpearmanSignificant",
    "de_overlap_metric",
    "DESpearmanLFC",
    "compute_pr_auc",
    "compute_roc_auc",
    "DESigGenesRecall",
    "DENsigCounts",
    # Global registry
    "metrics_registry",
    # Base Classes
    "Metric",
    "MetricResult",
    "MetricInfo",
]
