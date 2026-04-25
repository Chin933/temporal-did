from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StrategyResult:
    name: str
    estimate: float
    std_error: float
    n_treated: int
    n_control: int
    notes: str = ""

    @property
    def ci_95(self) -> Tuple[float, float]:
        return (
            self.estimate - 1.96 * self.std_error,
            self.estimate + 1.96 * self.std_error,
        )

    @property
    def t_stat(self) -> float:
        return self.estimate / self.std_error if self.std_error > 0 else float("nan")

    @property
    def p_value(self) -> float:
        if np.isnan(self.t_stat):
            return float("nan")
        df = self.n_treated + self.n_control - 2
        return float(2 * (1 - stats.t.cdf(abs(self.t_stat), df=df)))


@dataclass
class DiagnosticsOutput:
    shock_year: int
    pre_year: int
    post_year: int
    tau: float
    mismatch_severity: float
    strategies: Dict[str, StrategyResult]
    sensitivity: pd.DataFrame

    def summary(self) -> str:
        sev = self.mismatch_severity
        sev_label = "HIGH" if sev > 0.75 else ("MODERATE" if sev > 0.4 else "LOW")
        pos_label = "closer to t2" if self.tau > 0.5 else "closer to t1"
        lines = [
            "Timing Mismatch Diagnostics",
            f"  Cross-sections : t1={self.pre_year}, t2={self.post_year}",
            f"  Shock year     : t*={self.shock_year}",
            f"  Position tau   : {self.tau:.3f}  ({pos_label})",
            f"  Mismatch sev.  : {self.mismatch_severity:.3f}  ({sev_label})",
            "",
            f"  {'Strategy':<18} {'Estimate':>10} {'SE':>8} {'95% CI':>22}  Notes",
            "  " + "-" * 85,
        ]
        for res in self.strategies.values():
            lo, hi = res.ci_95
            lines.append(
                f"  {res.name:<18} {res.estimate:>10.4f} {res.std_error:>8.4f}"
                f"  [{lo:>8.4f}, {hi:>8.4f}]  {res.notes}"
            )
        return "\n".join(lines)


def _compute_did(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    unit_id: str,
    year_col: str,
    pre_year: int,
    post_year: int,
) -> Tuple[float, float, int, int]:
    pre = data[data[year_col] == pre_year].set_index(unit_id)[[outcome, treatment]]
    post = data[data[year_col] == post_year].set_index(unit_id)[[outcome]]
    post.columns = [f"{outcome}_post"]

    panel = pre.join(post).dropna()
    panel["delta"] = panel[f"{outcome}_post"] - panel[outcome]

    treated = panel.loc[panel[treatment] == 1, "delta"]
    control = panel.loc[panel[treatment] == 0, "delta"]

    if len(treated) < 2 or len(control) < 2:
        raise ValueError("Too few observations in treatment or control group.")

    est = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control)))
    return est, se, int(len(treated)), int(len(control))


def timing_mismatch_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    shock_year: int,
    pre_year: int,
    post_year: int,
    unit_id: str,
    year_col: str = "year",
    adjustment_rho: float = 0.95,
    decay_rates: Optional[List[float]] = None,
) -> DiagnosticsOutput:
    """
    Diagnose timing mismatch in historical difference-in-differences.

    When a policy shock at t* falls between cross-sections t1 < t* < t2,
    standard DiD estimates ATT(t2) rather than the target ATT(t*). This
    function measures mismatch severity and provides estimates under three
    assumptions about treatment effect dynamics.

    Parameters
    ----------
    data : DataFrame
        Panel data with columns for unit identifier, year, outcome, treatment.
    outcome : str
        Outcome variable column name.
    treatment : str
        Binary treatment indicator (0/1) column name.
    shock_year : int
        Year of the policy shock (t*). Must satisfy pre_year < shock_year < post_year.
    pre_year : int
        Year of the pre-treatment cross-section (t1).
    post_year : int
        Year of the post-treatment cross-section (t2).
    unit_id : str
        Unit identifier column name.
    year_col : str
        Year column name (default: "year").
    adjustment_rho : float
        Per-year decay rate for the AR(1)-adjusted estimate (default: 0.95).
    decay_rates : list of float, optional
        Grid of rho values for sensitivity analysis. Default: linspace(0.5, 1.5, 100).

    Returns
    -------
    DiagnosticsOutput

    Notes
    -----
    Mismatch severity = 4*tau*(1-tau), tau = (t*-t1)/(t2-t1).
    Equals 1 when the shock falls midway between cross-sections, 0 when
    it coincides with a cross-section year.

    Strategies
    ----------
    standard     : DiD(t1, t2). Estimates ATT(t2); no correction.
    ar1_adjusted : DiD / rho^(t2-t*). Recovers ATT(t*) under AR(1) dynamics.
    monotone_lb  : DiD(t1, t2) as lower bound on ATT(t*) when treatment
                   effects are non-increasing over time.
    """
    if not (pre_year < shock_year < post_year):
        raise ValueError(
            f"Require pre_year < shock_year < post_year; "
            f"got {pre_year} < {shock_year} < {post_year}."
        )

    tau = (shock_year - pre_year) / (post_year - pre_year)
    severity = 4 * tau * (1 - tau)
    gap = post_year - shock_year

    est, se, n_t, n_c = _compute_did(
        data, outcome, treatment, unit_id, year_col, pre_year, post_year
    )

    adj = adjustment_rho**gap
    strategies: Dict[str, StrategyResult] = {
        "standard": StrategyResult(
            name="standard",
            estimate=est,
            std_error=se,
            n_treated=n_t,
            n_control=n_c,
            notes=f"Estimates ATT(t2={post_year}); may under-estimate ATT(t*) if effects decay",
        ),
        "ar1_adjusted": StrategyResult(
            name="ar1_adjusted",
            estimate=est / adj,
            std_error=se / adj,
            n_treated=n_t,
            n_control=n_c,
            notes=f"ATT(t*) under AR(1) decay rho={adjustment_rho}; check sensitivity plot",
        ),
        "monotone_lb": StrategyResult(
            name="monotone_lb",
            estimate=est,
            std_error=se,
            n_treated=n_t,
            n_control=n_c,
            notes="Lower bound on ATT(t*) assuming non-increasing treatment effects",
        ),
    }

    if decay_rates is None:
        decay_rates = list(np.linspace(0.5, 1.5, 100))

    sensitivity = pd.DataFrame(
        {
            "decay_rate": decay_rates,
            "implied_att": [est / (rho**gap) for rho in decay_rates],
            "standard_did": est,
            "gap_years": gap,
        }
    )

    return DiagnosticsOutput(
        shock_year=shock_year,
        pre_year=pre_year,
        post_year=post_year,
        tau=tau,
        mismatch_severity=severity,
        strategies=strategies,
        sensitivity=sensitivity,
    )
