__version__ = "0.2.0"

from .classify import CaseClassification, PostDiD
from .diagnostics import (
    DiagnosticsOutput,
    StrategyResult,
    TemporalMismatchResult,
    diagnose,
    timing_mismatch_diagnostics,
)
from .plot import (
    plot_case_diagram,
    plot_diagnostics,
    plot_identified_set,
    plot_monte_carlo,
    plot_post_dids,
    plot_sensitivity,
    plot_strategy_comparison,
    plot_temporal_mismatch,
)

__all__ = [
    "__version__",
    # New multi-period API
    "diagnose",
    "TemporalMismatchResult",
    "CaseClassification",
    "PostDiD",
    "plot_temporal_mismatch",
    "plot_post_dids",
    "plot_identified_set",
    "plot_case_diagram",
    # Legacy single-pair API
    "timing_mismatch_diagnostics",
    "DiagnosticsOutput",
    "StrategyResult",
    "plot_diagnostics",
    "plot_sensitivity",
    "plot_strategy_comparison",
    "plot_monte_carlo",
]
