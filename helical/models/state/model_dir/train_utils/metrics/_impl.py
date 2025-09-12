from .._types import MetricBestValue, MetricType
from ._anndata import (
    ClusteringAgreement,
    discrimination_score,
    edistance,
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
from ._registry import MetricRegistry

metrics_registry = MetricRegistry()

metrics_registry.register(
    name="pearson_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Pearson correlation between mean differences from control",
    best_value=MetricBestValue.ONE,
    func=pearson_delta,
)

metrics_registry.register(
    name="mse",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation from control.",
    best_value=MetricBestValue.ZERO,
    func=mse,
)
metrics_registry.register(
    name="mae",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean absolute error of each perturbation from control.",
    best_value=MetricBestValue.ZERO,
    func=mae,
)

metrics_registry.register(
    name="mse_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
    best_value=MetricBestValue.ZERO,
    func=mse_delta,
)

metrics_registry.register(
    name="mae_delta",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Mean squared error of each perturbation-control delta.",
    best_value=MetricBestValue.ZERO,
    func=mae_delta,
)

for distance_metric in ["l1", "l2", "cosine"]:
    metrics_registry.register(
        name=f"discrimination_score_{distance_metric}",
        metric_type=MetricType.ANNDATA_PAIR,
        description=f"Determines similarity of each pred representation to real via normalized rank: {distance_metric}",
        best_value=MetricBestValue.ONE,
        func=discrimination_score,
        kwargs={"metric": distance_metric},
    )

metrics_registry.register(
    name="pearson_edistance",
    metric_type=MetricType.ANNDATA_PAIR,
    best_value=MetricBestValue.ONE,
    description="Calculates the pearson correlation coefficient between all pred and real edistance from controls",
    func=edistance,
)


for metric in ["overlap", "precision"]:
    for n in [None, 50, 100, 200, 500]:
        repr = n if n else "N"
        metrics_registry.register(
            name=f"{metric}_at_{repr}",
            metric_type=MetricType.DE,
            description=f"Overlap metric ({metric}) of top {repr} DE genes",
            best_value=MetricBestValue.ONE,
            func=de_overlap_metric,
            kwargs={"k": n, "metric": metric},
        )


metrics_registry.register(
    name="de_spearman_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation on number of significant DE genes",
    best_value=MetricBestValue.ONE,
    func=DESpearmanSignificant,  # type: ignore
    is_class=True,
)

metrics_registry.register(
    name="de_direction_match",
    metric_type=MetricType.DE,
    description="Agreement in direction of DE gene changes",
    best_value=MetricBestValue.ONE,
    func=DEDirectionMatch,  # type: ignore
    is_class=True,
)

metrics_registry.register(
    name="de_spearman_lfc_sig",
    metric_type=MetricType.DE,
    description="Spearman correlation on log fold changes of significant genes",
    best_value=MetricBestValue.ONE,
    func=DESpearmanLFC,  # type: ignore
    is_class=True,
)

metrics_registry.register(
    name="de_sig_genes_recall",
    metric_type=MetricType.DE,
    description="Recall of significant genes",
    best_value=MetricBestValue.ONE,
    func=DESigGenesRecall,  # type: ignore
    is_class=True,
)

metrics_registry.register(
    name="de_nsig_counts",
    metric_type=MetricType.DE,
    description="Counts of significant genes",
    best_value=MetricBestValue.NONE,
    func=DENsigCounts,  # type: ignore
    is_class=True,
)

metrics_registry.register(
    name="pr_auc",
    metric_type=MetricType.DE,
    description="Computes precision-recall for significant recovery",
    best_value=MetricBestValue.ONE,
    func=compute_pr_auc,
)

metrics_registry.register(
    name="roc_auc",
    metric_type=MetricType.DE,
    description="Computes ROC AUC for significant recovery",
    best_value=MetricBestValue.ONE,
    func=compute_roc_auc,
)

metrics_registry.register(
    name="clustering_agreement",
    metric_type=MetricType.ANNDATA_PAIR,
    description="Clustering agreement between real and predicted perturbation centroids",
    best_value=MetricBestValue.ONE,
    func=ClusteringAgreement,  # type: ignore
    is_class=True,
)
