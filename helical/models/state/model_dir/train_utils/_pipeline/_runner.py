import logging
from typing import Any, Callable, Literal

import polars as pl

from .._types import DEComparison, MetricBestValue, MetricType, PerturbationAnndataPair
from ..metrics import MetricResult, metrics_registry

logger = logging.getLogger(__name__)

MINIMAL_METRICS = [
    "pearson_delta",
    "mse",
    "mae",
    "discrimination_score_l1",
    "overlap_at_N",
    "precision_at_N",
    "de_nsig_counts",
]

VCC_METRICS = [
    "mae",
    "discrimination_score_l1",
    "overlap_at_N",
]

KNOWN_PROFILES = [
    "full",
    "minimal",
    "vcc",
    "de",
    "anndata",
]


class MetricPipeline:
    """Pipeline for computing metrics."""

    def __init__(
        self,
        profile: Literal["full", "minimal", "vcc", "de", "anndata"] | None = "full",
        metric_configs: dict[str, dict[str, Any]] | None = None,
        break_on_error: bool = False,
    ) -> None:
        """Initialize pipeline.

        Args:
            profile: Which set of metrics to compute ('full', 'de', 'anndata', or None)
            metric_configs: Dictionary mapping metric names to their configuration kwargs
            break_on_error: Whether to stop the pipeline on error
        """
        self._metrics: list[str] = []
        self._results: list[MetricResult] = []
        self._metric_configs = metric_configs or {}
        self._break_on_error = break_on_error
        self._results_df = None

        match profile:
            case "full":
                self._metrics.extend(metrics_registry.list_metrics(MetricType.DE))
                self._metrics.extend(
                    metrics_registry.list_metrics(MetricType.ANNDATA_PAIR)
                )
            case "de":
                self._metrics.extend(metrics_registry.list_metrics(MetricType.DE))
            case "anndata":
                self._metrics.extend(
                    metrics_registry.list_metrics(MetricType.ANNDATA_PAIR)
                )
            case "minimal":
                self._metrics.extend(MINIMAL_METRICS)
            case "vcc":
                self._metrics.extend(VCC_METRICS)
            case None:
                pass
            case _:
                raise ValueError(f"Unrecognized profile: {profile}")

        # Apply metric configurations
        for metric_name, config in self._metric_configs.items():
            if metric_name in metrics_registry.list_metrics():
                metrics_registry.update_metric_kwargs(metric_name, config)

    def add_metrics(
        self, metrics: list[str], configs: dict[str, dict[str, Any]] | None = None
    ) -> None:
        """Add metrics to pipeline.

        Args:
            metrics: List of metric names to add
            configs: Optional dictionary mapping metric names to their configuration kwargs
        """
        self._metrics.extend(metrics)
        if configs:
            self._metric_configs.update(configs)
            for metric_name, config in configs.items():
                if metric_name in metrics_registry.list_metrics():
                    metrics_registry.update_metric_kwargs(metric_name, config)

    def add_metric(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        func: Callable[
            [PerturbationAnndataPair | DEComparison], float | dict[str, float]
        ],
        best_value: MetricBestValue,
        is_class: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Register a new metric and add it to the pipeline.

        Args:
            name: Unique name for the metric
            metric_type: Type of metric being registered
            description: Description of what the metric computes
            func: Function to compute the metric
            is_class: Whether the metric is a class that needs instantiation
            kwargs: Optional keyword arguments for the metric
        """
        if name in metrics_registry.list_metrics():
            logger.warning(
                f"Metric '{name}' already registered, skipping re-registering"
            )
            return
        # Register the metric with the registry
        metrics_registry.register(
            name=name,
            metric_type=metric_type,
            description=description,
            func=func,
            is_class=is_class,
            best_value=best_value,
            kwargs=kwargs,
        )
        # Add it to the pipeline's metrics list
        self._metrics.append(name)
        # Add any kwargs to the metric configs
        if kwargs:
            self._metric_configs[name] = kwargs

    def skip_metrics(
        self,
        to_skip: list[str] | str,
    ):
        if isinstance(to_skip, str):
            to_skip = [to_skip]
        self._metrics = [metric for metric in self._metrics if metric not in to_skip]

    def _compute_metric(
        self,
        name: str,
        data: DEComparison | PerturbationAnndataPair,
    ):
        """Compute a specific metric."""
        try:
            logger.info(f"Computing metric '{name}'")
            # Get any runtime config for this metric
            runtime_config = self._metric_configs.get(name, {})
            value = metrics_registry.compute(name, data, kwargs=runtime_config)
            if isinstance(value, dict):
                # Add each perturbation result separately
                for pert, pert_value in value.items():
                    # if the return value is a dictionary add the sub-metric
                    # as a separate result
                    if isinstance(pert_value, dict):
                        for sub_name, value in pert_value.items():
                            self._results.append(
                                MetricResult(
                                    name=f"{name}_{sub_name}",
                                    value=value,
                                    perturbation=pert,
                                )
                            )
                    else:
                        self._results.append(
                            MetricResult(
                                name=name,
                                value=pert_value,
                                perturbation=pert,
                            )
                        )
            else:
                # Add single result to all perturbations
                for pert in data.get_perts(include_control=False):
                    self._results.append(
                        MetricResult(
                            name=name,
                            value=value,
                            perturbation=pert,
                        )
                    )
        except Exception as error:
            logger.error(f"Error computing metric '{name}': {error}")
            if self._break_on_error:
                raise error

    def compute_de_metrics(self, data: DEComparison) -> None:
        """Compute DE metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.DE):
                continue
            self._compute_metric(name, data)

    def compute_anndata_metrics(
        self,
        data: PerturbationAnndataPair,
    ) -> None:
        """Compute perturbation metrics."""
        for name in self._metrics:
            if name not in metrics_registry.list_metrics(MetricType.ANNDATA_PAIR):
                continue
            self._compute_metric(name, data)

    def get_results(self) -> pl.DataFrame:
        """Get results as a DataFrame."""
        if self._results_df is None:
            self._results_df = (
                pl.DataFrame()
                if not self._results
                else pl.DataFrame([r.to_dict() for r in self._results]).pivot(
                    index="perturbation",
                    on="metric",
                    values="value",
                )
            )
        return self._results_df

    def get_agg_results(self) -> pl.DataFrame:
        """Get aggregated results as a DataFrame."""
        return self.get_results().drop("perturbation").describe()
