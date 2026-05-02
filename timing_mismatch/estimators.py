from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .classify import PostDiD
from .diagnostics import StrategyResult, _compute_did


# ---------------------------------------------------------------------------
# Case 1: effect persists / grows
# ---------------------------------------------------------------------------


def estimate_case1(post_dids: Dict[int, PostDiD]) -> Dict[str, StrategyResult]:
    """Case 1: Report dynamic ATT(tk) for each post-shock cross-section."""
    strategies: Dict[str, StrategyResult] = {}
    for yr, d in sorted(post_dids.items()):
        key = f"att_{yr}"
        strategies[key] = StrategyResult(
            name=key,
            estimate=d.estimate,
            std_error=d.std_error,
            n_treated=d.n_treated,
            n_control=d.n_control,
            notes=f"ATT(t={yr}); {d.gap} yrs post-shock",
        )
    return strategies


# ---------------------------------------------------------------------------
# Case 2: decaying effect — joint log-linear estimation of rho and ATT(t*)
# ---------------------------------------------------------------------------
#
# Model (AR-1):  DiD_k = ATT(t*) * rho^(t_k - t*)
# Log-linearise: log|DiD_k| = log|ATT(t*)| + (t_k - t*) * log(rho)
# Fit via WLS (weights = 1 / SE²) across all available post periods or
# cohort × calendar-time cells (staggered variant).


def _fit_log_linear(
    event_gaps: np.ndarray,
    log_abs_dids: np.ndarray,
    weights: np.ndarray,
    sign: float,
    n_treated_mean: int,
    n_control_mean: int,
    label_prefix: str,
) -> Tuple[Dict[str, StrategyResult], float]:
    """Shared WLS fitter for Case 2 (multi-period and staggered)."""
    X = np.column_stack([np.ones(len(event_gaps)), event_gaps])
    W = np.diag(weights)
    XtW = X.T @ W
    beta = np.linalg.solve(XtW @ X, XtW @ log_abs_dids)

    resid = log_abs_dids - X @ beta
    sigma2 = float(np.dot(resid * weights, resid) / max(len(event_gaps) - 2, 1))
    cov = sigma2 * np.linalg.inv(XtW @ X)

    log_att, log_rho = float(beta[0]), float(beta[1])
    se_log_att = float(np.sqrt(max(cov[0, 0], 0.0)))
    se_log_rho = float(np.sqrt(max(cov[1, 1], 0.0)))

    att_star = np.exp(log_att) * sign
    se_att = abs(att_star) * se_log_att
    rho_hat = np.exp(log_rho)
    se_rho = rho_hat * se_log_rho

    strategies: Dict[str, StrategyResult] = {
        f"{label_prefix}_att_star": StrategyResult(
            name=f"{label_prefix}_att_star",
            estimate=float(att_star),
            std_error=float(se_att),
            n_treated=n_treated_mean,
            n_control=n_control_mean,
            notes=(
                f"ATT(t*) via joint log-linear; "
                f"rho_hat={rho_hat:.3f} (SE={se_rho:.3f})"
            ),
        ),
        f"{label_prefix}_rho": StrategyResult(
            name=f"{label_prefix}_rho",
            estimate=float(rho_hat),
            std_error=float(se_rho),
            n_treated=n_treated_mean,
            n_control=n_control_mean,
            notes=f"Estimated per-year decay rate; log(rho)={log_rho:.4f}±{se_log_rho:.4f}",
        ),
    }
    return strategies, float(rho_hat)


def estimate_case2_multiperiod(
    post_dids: Dict[int, PostDiD],
    shock_year: int,
) -> Tuple[Dict[str, StrategyResult], float]:
    """
    Case 2 (single shock, multiple post periods): jointly estimate rho and ATT(t*).

    Requires >= 2 post periods where |DiD| > 0.
    """
    valid = [
        (yr, d)
        for yr, d in sorted(post_dids.items())
        if abs(d.estimate) > 1e-10 and d.std_error > 0
    ]
    if len(valid) < 2:
        raise ValueError(
            "Joint estimation requires >= 2 post periods with detectable DiD. "
            "Consider sensitivity analysis (Case 3a) instead."
        )

    gaps = np.array([yr - shock_year for yr, _ in valid], dtype=float)
    log_abs = np.array([np.log(abs(d.estimate)) for _, d in valid])
    w = np.array([1.0 / d.std_error**2 for _, d in valid])
    sign = float(np.sign(valid[0][1].estimate))
    n_t = int(round(np.mean([d.n_treated for _, d in valid])))
    n_c = int(round(np.mean([d.n_control for _, d in valid])))

    return _fit_log_linear(gaps, log_abs, w, sign, n_t, n_c, "joint")


def estimate_case2_staggered(
    cohort_dids: Dict[Tuple[int, int], PostDiD],
) -> Tuple[Dict[str, StrategyResult], float]:
    """
    Case 2 (staggered): jointly estimate rho and ATT(t*) using cohort variation.

    cohort_dids maps (cohort_year, calendar_year) -> PostDiD.
    PostDiD.gap = calendar_year - cohort_year (event time).

    Assumes homogeneous ATT(t*) across cohorts:
        log|DiD(g, t)| = log|ATT(t*)| + (t - g) * log(rho)
    """
    valid = [(key, d) for key, d in cohort_dids.items() if abs(d.estimate) > 1e-10 and d.std_error > 0]
    if len(valid) < 2:
        raise ValueError(
            "Staggered joint estimation requires >= 2 cohort-period cells with detectable DiD."
        )

    event_times = np.array([d.gap for _, d in valid], dtype=float)
    log_abs = np.array([np.log(abs(d.estimate)) for _, d in valid])
    w = np.array([1.0 / d.std_error**2 for _, d in valid])
    sign = float(np.sign(valid[0][1].estimate))
    n_t = int(round(np.mean([d.n_treated for _, d in valid])))
    n_c = int(round(np.mean([d.n_control for _, d in valid])))

    return _fit_log_linear(event_times, log_abs, w, sign, n_t, n_c, "staggered")


# ---------------------------------------------------------------------------
# Case 3a: all post DiDs ~ 0 — sensitivity / identified set
# ---------------------------------------------------------------------------


def compute_identified_set(
    post_dids: Dict[int, PostDiD],
    shock_year: int,
    rho_grid: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Case 3a: For each assumed rho, back out the implied ATT(t*) and its 95% CI.

    Under AR(1): ATT(t*) = DiD_k / rho^(t_k - t*)

    Uses the earliest available post period (smallest gap) for the tightest
    bounds. A very small rho makes the upper bound large — the plot shows
    how much ATT(t*) could be hidden by rapid decay.

    Columns
    -------
    rho, point_att_star, ci_lower, ci_upper, min_detectable, gap_years, post_year
    """
    if rho_grid is None:
        rho_grid = list(np.linspace(0.05, 1.0, 200))

    earliest = min(post_dids.keys())
    d = post_dids[earliest]
    gap = earliest - shock_year
    lb_did, ub_did = d.ci_95

    rows = []
    for rho in rho_grid:
        if rho <= 0:
            continue
        adj = rho**gap
        rows.append(
            {
                "rho": rho,
                "point_att_star": d.estimate / adj,
                "ci_lower": lb_did / adj,
                "ci_upper": ub_did / adj,
                "min_detectable": 1.96 * d.std_error / adj,
                "gap_years": gap,
                "post_year": earliest,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: cohort DiD for staggered data
# ---------------------------------------------------------------------------


def compute_cohort_did(
    data: pd.DataFrame,
    outcome: str,
    unit_id: str,
    year_col: str,
    shock_year_col: str,
    cohort_year: int,
    pre_year: int,
    post_year: int,
    shock_year: int,
) -> PostDiD:
    """
    DiD for cohort g vs never-treated units at (pre_year, post_year).

    Never-treated units are identified by NaN or 0 in shock_year_col.
    """
    never_mask = data[shock_year_col].isna() | (data[shock_year_col] == 0)
    cohort_mask = data[shock_year_col] == cohort_year

    panel = data[
        data[year_col].isin([pre_year, post_year]) & (cohort_mask | never_mask)
    ].copy()

    pre = panel[panel[year_col] == pre_year].set_index(unit_id)[
        [outcome, shock_year_col]
    ]
    post_df = panel[panel[year_col] == post_year].set_index(unit_id)[[outcome]]
    post_df.columns = [f"{outcome}_post"]

    merged = pre.join(post_df).dropna()
    merged["delta"] = merged[f"{outcome}_post"] - merged[outcome]

    treated = merged.loc[merged[shock_year_col] == cohort_year, "delta"]
    ctrl_idx = merged.index[never_mask.reindex(merged.index, fill_value=False)]
    control = merged.loc[ctrl_idx, "delta"]

    if len(treated) < 2 or len(control) < 2:
        raise ValueError(
            f"Too few observations for cohort {cohort_year} at period {post_year}."
        )

    est = float(treated.mean() - control.mean())
    se = float(np.sqrt(treated.var() / len(treated) + control.var() / len(control)))

    return PostDiD(
        year=post_year,
        estimate=est,
        std_error=se,
        n_treated=int(len(treated)),
        n_control=int(len(control)),
        gap=post_year - cohort_year,
    )
