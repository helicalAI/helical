from typing import Any, Callable, Dict, List, Optional

from .._types import DEComparison, MetricBestValue, MetricType, PerturbationAnndataPair
from .base import MetricInfo

METRIC_FUNC_ADATA_KWARGS = Callable[
    [PerturbationAnndataPair, Any], float | dict[str, float]
]
METRIC_FUNC_ADATA = Callable[[PerturbationAnndataPair], float | dict[str, float]]
METRIC_FUNC_DE_KWARGS = Callable[[DEComparison, Any], float | dict[str, float]]
METRIC_FUNC_DE = Callable[[DEComparison], float | dict[str, float]]

METRIC_FUNC = (
    METRIC_FUNC_ADATA
    | METRIC_FUNC_DE
    | METRIC_FUNC_DE_KWARGS
    | METRIC_FUNC_ADATA_KWARGS
)


class MetricRegistry:
    """Registry for managing and accessing metrics."""

    def __init__(self) -> None:
        self.metrics: Dict[str, MetricInfo] = {}

    def register(
        self,
        name: str,
        metric_type: MetricType,
        description: str,
        func: METRIC_FUNC,
        best_value: MetricBestValue,
        is_class: bool = False,
        kwargs: dict[str, Any] | None = None,
    ):
        """
        Register a new metric.

        Args:
            name: Unique name for the metric
            metric_type: Type of metric being registered
            description: Description of what the metric computes
            func: Function to compute the metric
            best_value: Best value for the metric
            is_class: Whether the metric is a class that needs instantiation
            kwargs: Optional keyword arguments for the metric
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already registered")
        self.metrics[name] = MetricInfo(
            name=name,
            type=metric_type,
            func=func,
            description=description,
            best_value=best_value,
            is_class=is_class,
            kwargs=kwargs,
        )

    def update_metric_kwargs(self, name: str, kwargs: dict[str, Any]) -> None:
        """
        Update the keyword arguments for a registered metric.

        Args:
            name: Name of the metric to update
            kwargs: New keyword arguments to use
        """
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in registry")

        if self.metrics[name].kwargs is None:
            self.metrics[name].kwargs = {}

        self.metrics[name].kwargs.update(kwargs)  # type: ignore

    def get_metric(self, name: str) -> MetricInfo:
        """Get information about a registered metric."""
        if name not in self.metrics:
            raise KeyError(f"Metric '{name}' not found in registry")
        return self.metrics[name]

    def list_metrics(self, metric_type: Optional[MetricType] = None) -> List[str]:
        """
        List registered metrics, optionally filtered by type.

        Args:
            metric_type: If provided, only list metrics of this type

        Returns:
            List of metric names
        """
        if metric_type is None:
            return list(self.metrics.keys())
        return [name for name, info in self.metrics.items() if info.type == metric_type]

    def compute(
        self,
        name: str,
        data: PerturbationAnndataPair | DEComparison,
        kwargs: dict[str, Any] | None = None,
    ) -> float | dict[str, float]:
        """
        Compute a metric on the provided data.

        Args:
            name: Name of the metric to compute
            data: Data to compute the metric on
            kwargs: Optional keyword arguments to override stored kwargs

        Returns:
            Metric result, either a single float or dictionary of values
        """
        metric = self.get_metric(name)
        # Merge stored kwargs with any provided kwargs
        merged_kwargs = metric.kwargs.copy()  # type: ignore
        if kwargs:
            merged_kwargs.update(kwargs)

        if metric.is_class:
            # Instantiate the class before calling
            instance = metric.func(**merged_kwargs)
            return instance(data)
        return metric.func(data, **merged_kwargs)
