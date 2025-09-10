import enum


class DESortBy(enum.Enum):
    """Sorting options for differential expression results."""

    FOLD_CHANGE = "log2_fold_change"
    ABS_FOLD_CHANGE = "abs_log2_fold_change"
    PVALUE = "p_value"
    FDR = "fdr"


class MetricType(enum.Enum):
    """Types of metrics supported by the registry."""

    DE = "de"
    ANNDATA_PAIR = "anndata_pair"


class MetricBestValue(enum.Enum):
    """How to comparse this metric w.r.t the baseline.

    Design your metrics to be bounded between 0 and 1.

    If the metric's best value is zero, set this to `ZERO`
    If the metric's best value is one, set this to `ONE`
    If the metric should not be compared to baseline, set this to `NONE`
    """

    ZERO = "zero"
    ONE = "one"
    NONE = "none"
