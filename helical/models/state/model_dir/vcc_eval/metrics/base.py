from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .._types import DEComparison, MetricBestValue, MetricType, PerturbationAnndataPair


class Metric(Protocol):
    """Protocol for metric functions."""

    def __call__(
        self,
        data: PerturbationAnndataPair | DEComparison,
        **kwargs,
    ) -> float | dict[str, float]: ...


@dataclass
class MetricResult:
    """Result of a metric computation."""

    name: str
    value: float | str
    perturbation: str | None = None

    def to_dict(self) -> dict[str, float | str]:
        """Convert result to dictionary."""
        return {
            "perturbation": self.perturbation,  # type: ignore
            "metric": self.name,
            "value": self.value,
        }


@dataclass
class MetricInfo:
    """Information about a registered metric."""

    name: str
    type: MetricType
    func: Callable
    description: str
    best_value: MetricBestValue
    is_class: bool = False
    kwargs: dict[str, Any] | None = None

    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
