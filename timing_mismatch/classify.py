from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Tuple

import numpy as np
from scipy import stats

CaseType = Literal["1", "2", "3a", "3b"]


@dataclass
class PostDiD:
    """DiD estimate for a single post-shock cross-section."""

    year: int
    estimate: float
    std_error: float
    n_treated: int
    n_control: int
    gap: int  # calendar years since shock (year - shock_year)

    @property
    def t_stat(self) -> float:
        return self.estimate / self.std_error if self.std_error > 0 else float("nan")

    @property
    def p_value(self) -> float:
        if np.isnan(self.t_stat):
            return float("nan")
        df = self.n_treated + self.n_control - 2
        return float(2 * (1 - stats.t.cdf(abs(self.t_stat), df=df)))

    @property
    def ci_95(self) -> Tuple[float, float]:
        return (
            self.estimate - 1.96 * self.std_error,
            self.estimate + 1.96 * self.std_error,
        )


@dataclass
class CaseClassification:
    """Output of the case-classification step."""

    case: CaseType
    description: str
    has_staggered: bool
    n_significant: int
    n_post: int
    alpha: float


def classify_case(
    post_dids: Dict[int, PostDiD],
    has_staggered: bool,
    alpha: float = 0.10,
) -> CaseClassification:
    """
    Classify the temporal mismatch setting into one of four cases.

    Cases
    -----
    1  : Effect persists or grows — all/most post DiDs significant, no decline.
         Standard DiD valid; report dynamic ATT(tk).
    2  : Effect decays but visible — significant decline across post periods,
         or staggered timing available. Jointly estimate rho and ATT(t*).
    3a : No post period shows a significant effect and no staggered timing.
         Data cannot identify ATT(t*); sensitivity bounds only.
    3b : No post period shows a significant effect but staggered timing available.
         Use cohort variation (same logic as Case 2) to attempt identification.

    Parameters
    ----------
    post_dids : dict
        Mapping post_year -> PostDiD for all available post-shock cross-sections.
    has_staggered : bool
        Whether the data contains staggered treatment timing (unit-level shock years).
    alpha : float
        Significance threshold for determining effect visibility (default 0.10).
    """
    n_sig = sum(1 for d in post_dids.values() if d.p_value < alpha)
    n_post = len(post_dids)

    if n_sig > 0:
        years = sorted(post_dids.keys())
        estimates = [post_dids[y].estimate for y in years]

        # Detect monotone decline: mean(effect) and time-slope have opposite signs
        declining = False
        if len(years) >= 2:
            slope = float(np.polyfit(range(len(years)), estimates, 1)[0])
            mean_est = float(np.mean(estimates))
            declining = (mean_est > 0 and slope < 0) or (mean_est < 0 and slope > 0)

        # Case 2 needs either >=2 post periods (multi-period regression) or staggered
        can_estimate_rho = (n_post >= 2) or has_staggered
        if declining and can_estimate_rho:
            case: CaseType = "2"
            desc = (
                "Effect decays but visible in post periods — "
                "jointly estimate rho and ATT(t*) via log-linear regression"
            )
        else:
            case = "1"
            desc = (
                "Effect persists or grows — "
                "standard DiD valid; dynamic ATT(tk) reported for each post period"
            )
    else:
        if has_staggered:
            case = "3b"
            desc = (
                "No significant effect in any post period (staggered data available) — "
                "use cohort timing variation to attempt ATT(t*) identification"
            )
        else:
            case = "3a"
            desc = (
                "No significant effect in any post period — "
                "ATT(t*) is not identified; sensitivity bounds over rho reported"
            )

    return CaseClassification(
        case=case,
        description=desc,
        has_staggered=has_staggered,
        n_significant=n_sig,
        n_post=n_post,
        alpha=alpha,
    )
