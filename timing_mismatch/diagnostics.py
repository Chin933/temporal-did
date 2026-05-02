from __future__ import annotations

from dataclasses import dataclass, field
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


# ---------------------------------------------------------------------------
# New API: multi-period, case-aware diagnostics
# ---------------------------------------------------------------------------


@dataclass
class TemporalMismatchResult:
    """
    Output of the comprehensive temporal mismatch diagnostic.

    Attributes
    ----------
    shock_year : int
        Policy shock year t* (not observed in data).
    pre_years : list of int
        Available pre-shock cross-section years.
    post_years : list of int
        Available post-shock cross-section years.
    classification : CaseClassification
        Detected case (1 / 2 / 3a / 3b) and description.
    post_dids : dict
        post_year -> PostDiD for each post-shock period.
    strategies : dict
        Strategy name -> StrategyResult (case-dependent estimates).
    estimated_rho : float or None
        Estimated decay rate (Cases 2 / 3b only).
    identified_set : DataFrame or None
        Sensitivity bounds over rho (Cases 3a / 3b only).
    """

    shock_year: int
    pre_years: List[int]
    post_years: List[int]
    classification: "CaseClassification"  # noqa: F821
    post_dids: Dict[int, "PostDiD"]  # noqa: F821
    strategies: Dict[str, StrategyResult]
    estimated_rho: Optional[float]
    identified_set: Optional[pd.DataFrame]

    def summary(self) -> str:
        cls = self.classification
        lines = [
            "Temporal Mismatch Diagnostics",
            f"  Shock year   : t* = {self.shock_year}",
            f"  Pre periods  : {self.pre_years}",
            f"  Post periods : {self.post_years}",
            f"  Case         : {cls.case}  —  {cls.description}",
            f"  Staggered    : {'yes' if cls.has_staggered else 'no'}",
            f"  Significant post-DiDs : {cls.n_significant} / {cls.n_post}"
            f"  (alpha={cls.alpha})",
        ]
        if self.strategies:
            lines += [
                "",
                f"  {'Strategy':<22} {'Estimate':>10} {'SE':>8} {'95% CI':>22}  Notes",
                "  " + "-" * 90,
            ]
            for res in self.strategies.values():
                lo, hi = res.ci_95
                lines.append(
                    f"  {res.name:<22} {res.estimate:>10.4f} {res.std_error:>8.4f}"
                    f"  [{lo:>8.4f}, {hi:>8.4f}]  {res.notes}"
                )
        if self.identified_set is not None:
            for rho_target in [0.9, 0.7, 0.5]:
                row = self.identified_set.iloc[
                    (self.identified_set["rho"] - rho_target).abs().argmin()
                ]
                lines.append(
                    f"  (3a) rho={rho_target:.1f}: "
                    f"ATT(t*) in [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]"
                )
        return "\n".join(lines)


def diagnose(
    data: pd.DataFrame,
    outcome: str,
    treatment: str,
    shock_year: int,
    pre_years: List[int],
    post_years: List[int],
    unit_id: str,
    year_col: str = "year",
    shock_year_col: Optional[str] = None,
    alpha: float = 0.10,
    rho_grid: Optional[List[float]] = None,
) -> TemporalMismatchResult:
    """
    Comprehensive temporal mismatch diagnostic for multi-period panel data.

    Automatically classifies the setting and applies the appropriate estimator:

    - Case 1: Effect persists/grows — dynamic ATT(tk) for each post period.
    - Case 2: Effect decays but visible — joint log-linear estimation of
              rho and ATT(t*) using multiple post periods or staggered cohorts.
    - Case 3a: No post period shows an effect — identified-set bounds over rho.
    - Case 3b: No visible effect but staggered timing available — use cohort
               variation to attempt ATT(t*) identification.

    Parameters
    ----------
    data : DataFrame
        Panel data with cross-sections at years in pre_years | post_years.
    outcome, treatment, unit_id, year_col : str
        Column names for outcome variable, binary treatment, unit identifier,
        and year.
    shock_year : int
        Policy shock year t* (must not appear in pre_years or post_years).
    pre_years : list of int
        All available pre-shock cross-section years. The latest is used as
        the DiD baseline.
    post_years : list of int
        All available post-shock cross-section years.
    shock_year_col : str, optional
        Column containing each unit's treatment year for staggered settings.
        Never-treated units should have NaN or 0 in this column.
    alpha : float
        Significance threshold for case classification (default 0.10).
    rho_grid : list of float, optional
        Grid of rho values for Case 3a identified-set plot. Default: 200
        points on [0.05, 1.0].

    Returns
    -------
    TemporalMismatchResult
    """
    from .classify import CaseClassification, PostDiD, classify_case
    from .estimators import (
        compute_cohort_did,
        compute_identified_set,
        estimate_case1,
        estimate_case2_multiperiod,
        estimate_case2_staggered,
    )

    pre_baseline = max(pre_years)
    has_staggered = shock_year_col is not None

    # --- 1. Compute post-period DiDs (treatment vs control, baseline = latest pre) ---
    post_dids: Dict[int, PostDiD] = {}
    for yr in sorted(post_years):
        try:
            est, se, n_t, n_c = _compute_did(
                data, outcome, treatment, unit_id, year_col, pre_baseline, yr
            )
            post_dids[yr] = PostDiD(
                year=yr,
                estimate=est,
                std_error=se,
                n_treated=n_t,
                n_control=n_c,
                gap=yr - shock_year,
            )
        except ValueError:
            pass

    if not post_dids:
        raise ValueError(
            "No usable post-period DiD could be computed. "
            "Check that data contains observations at the specified post_years."
        )

    # --- 2. Classify case ---
    classification = classify_case(post_dids, has_staggered, alpha=alpha)
    case = classification.case

    # --- 3. Run case-appropriate estimator ---
    strategies: Dict[str, StrategyResult] = {}
    estimated_rho: Optional[float] = None
    identified_set: Optional[pd.DataFrame] = None

    if case == "1":
        strategies = estimate_case1(post_dids)

    elif case == "2":
        if has_staggered:
            cohort_dids = _build_cohort_dids(
                data, outcome, unit_id, year_col, shock_year_col,
                pre_years, post_years, shock_year,
            )
            if len(cohort_dids) >= 2:
                strategies, estimated_rho = estimate_case2_staggered(cohort_dids)
            else:
                strategies, estimated_rho = estimate_case2_multiperiod(
                    post_dids, shock_year
                )
        else:
            strategies, estimated_rho = estimate_case2_multiperiod(
                post_dids, shock_year
            )

    elif case in ("3a", "3b"):
        identified_set = compute_identified_set(post_dids, shock_year, rho_grid)
        if case == "3b" and has_staggered:
            cohort_dids = _build_cohort_dids(
                data, outcome, unit_id, year_col, shock_year_col,
                pre_years, post_years, shock_year,
            )
            if len(cohort_dids) >= 2:
                try:
                    strategies, estimated_rho = estimate_case2_staggered(cohort_dids)
                except ValueError:
                    pass

    return TemporalMismatchResult(
        shock_year=shock_year,
        pre_years=sorted(pre_years),
        post_years=sorted(post_years),
        classification=classification,
        post_dids=post_dids,
        strategies=strategies,
        estimated_rho=estimated_rho,
        identified_set=identified_set,
    )


def _build_cohort_dids(
    data: pd.DataFrame,
    outcome: str,
    unit_id: str,
    year_col: str,
    shock_year_col: str,
    pre_years: List[int],
    post_years: List[int],
    shock_year: int,
) -> Dict[Tuple[int, int], "PostDiD"]:
    """Compute cohort × calendar-time DiD table for staggered settings."""
    from .estimators import compute_cohort_did

    cohort_years = [
        int(c)
        for c in data[shock_year_col].dropna().unique()
        if c > 0 and c < max(post_years)
    ]

    cohort_dids: Dict[Tuple[int, int], "PostDiD"] = {}
    for g in cohort_years:
        valid_pre = [y for y in pre_years if y < g]
        if not valid_pre:
            continue
        pre_g = max(valid_pre)
        for t in [yr for yr in post_years if yr > g]:
            try:
                cd = compute_cohort_did(
                    data, outcome, unit_id, year_col,
                    shock_year_col, g, pre_g, t, shock_year,
                )
                cohort_dids[(g, t)] = cd
            except ValueError:
                pass
    return cohort_dids
