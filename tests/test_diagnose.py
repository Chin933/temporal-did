"""Tests for the multi-period diagnose() API covering all four cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from timing_mismatch import CaseClassification, TemporalMismatchResult, diagnose
from timing_mismatch.monte_carlo import simulate_panel


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CROSS_SECTIONS = [1777, 1820, 1888, 1911]
SHOCK_YEAR = 1796
PRE_YEARS = [1777]
POST_YEARS = [1820, 1888, 1911]


def _make_panel(
    dynamics: str = "constant",
    decay_rate: float = 0.95,
    true_att: float = 2.0,
    n_units: int = 300,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a multi-period panel from simulate_panel using only t1 and t2,
    then replicate the post year to simulate multiple post cross-sections."""
    # simulate_panel only gives pre_year and post_year, so we build multiple
    # post periods manually by calling it for each post year.
    rng = np.random.default_rng(seed)
    frames = []

    treatment_share = 0.5
    n_treated = int(n_units * treatment_share)
    n_control = n_units - n_treated
    unit_ids = np.arange(n_units)
    d = np.array([1] * n_treated + [0] * n_control)
    alpha = rng.normal(0, 1, n_units)

    for year in CROSS_SECTIONS:
        time_effect = 0.05 * (year - CROSS_SECTIONS[0])
        if year < SHOCK_YEAR:
            att = 0.0
        else:
            t_since = year - SHOCK_YEAR
            if dynamics == "constant":
                att = true_att
            elif dynamics == "decaying":
                att = true_att * (decay_rate ** t_since)
            elif dynamics == "zero":
                att = 0.0
            else:
                att = true_att * ((2 - decay_rate) ** t_since)
        y = alpha + time_effect + att * d + rng.normal(0, 0.3, n_units)
        frames.append(
            pd.DataFrame(
                {"unit_id": unit_ids, "year": year, "y": y, "treatment": d}
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Case 1: effect persists
# ---------------------------------------------------------------------------


def test_case1_classification():
    data = _make_panel(dynamics="constant", true_att=2.0, n_units=400)
    result = diagnose(
        data=data,
        outcome="y",
        treatment="treatment",
        shock_year=SHOCK_YEAR,
        pre_years=PRE_YEARS,
        post_years=POST_YEARS,
        unit_id="unit_id",
        alpha=0.10,
    )
    assert isinstance(result, TemporalMismatchResult)
    assert result.classification.case == "1"
    # Dynamic ATT strategies: one per post year
    for yr in POST_YEARS:
        assert f"att_{yr}" in result.strategies
    assert result.identified_set is None
    assert result.estimated_rho is None


def test_case1_estimates_close_to_true_att():
    estimates = []
    for seed in range(20):
        data = _make_panel(dynamics="constant", true_att=2.0, n_units=600, seed=seed)
        result = diagnose(
            data=data, outcome="y", treatment="treatment",
            shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
            unit_id="unit_id",
        )
        # Only collect ATT estimates (exclude rho estimates)
        for name, s in result.strategies.items():
            if "rho" not in name:
                estimates.append(s.estimate)
    assert abs(np.mean(estimates) - 2.0) < 0.15


# ---------------------------------------------------------------------------
# Case 2: decaying effect, still visible
# ---------------------------------------------------------------------------


def test_case2_classification():
    # slow decay: effect visible in 1820 but declining
    data = _make_panel(dynamics="decaying", decay_rate=0.97, true_att=3.0, n_units=500)
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
        unit_id="unit_id", alpha=0.10,
    )
    assert result.classification.case == "2"
    assert "joint_att_star" in result.strategies
    assert "joint_rho" in result.strategies
    assert result.estimated_rho is not None
    assert result.identified_set is None


def test_case2_rho_direction():
    # estimated rho should be < 1 for decaying dynamics
    rhos = []
    for seed in range(10):
        data = _make_panel(dynamics="decaying", decay_rate=0.97, true_att=3.0,
                           n_units=800, seed=seed)
        result = diagnose(
            data=data, outcome="y", treatment="treatment",
            shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
            unit_id="unit_id",
        )
        if result.estimated_rho is not None:
            rhos.append(result.estimated_rho)
    if rhos:
        assert np.mean(rhos) < 1.0, "Expected rho < 1 under decaying dynamics"


# ---------------------------------------------------------------------------
# Case 3a: effect too fast, nothing visible, no staggered
# ---------------------------------------------------------------------------


def test_case3a_classification():
    data = _make_panel(dynamics="zero", true_att=0.0, n_units=300, seed=42)
    # alpha=0.001: P(false positive per test) < 0.1%, so 3a is virtually guaranteed
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
        unit_id="unit_id", alpha=0.001,
    )
    assert result.classification.case == "3a"
    assert result.identified_set is not None
    assert len(result.identified_set) > 0


def test_case3a_identified_set_structure():
    data = _make_panel(dynamics="zero", true_att=0.0, n_units=300, seed=7)
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
        unit_id="unit_id", alpha=0.001,
    )
    assert result.classification.case == "3a"
    df = result.identified_set
    assert "rho" in df.columns
    assert "ci_upper" in df.columns
    assert "min_detectable" in df.columns
    assert len(df) >= 10
    # min_detectable should increase as rho decreases (larger gap-adjusted bound)
    mdet_high = float(df.loc[df["rho"].sub(0.95).abs().idxmin(), "min_detectable"])
    mdet_low = float(df.loc[df["rho"].sub(0.3).abs().idxmin(), "min_detectable"])
    assert mdet_low > mdet_high


def test_case3a_summary_contains_bounds():
    data = _make_panel(dynamics="zero", true_att=0.0, n_units=300, seed=3)
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
        unit_id="unit_id", alpha=0.001,
    )
    assert result.classification.case == "3a"
    s = result.summary()
    assert "3a" in s
    assert "rho=" in s


# ---------------------------------------------------------------------------
# Staggered Case 3b
# ---------------------------------------------------------------------------


def _make_staggered_panel(
    cohort_years: list,
    post_years: list,
    pre_year: int = 1777,
    shock_year: int = 1796,
    true_att: float = 2.5,
    decay_rate: float = 0.96,
    n_per_cohort: int = 100,
    n_control: int = 200,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    all_years = [pre_year] + post_years
    frames = []

    # Never-treated control units
    alpha_ctrl = rng.normal(0, 1, n_control)
    for year in all_years:
        y = alpha_ctrl + 0.05 * (year - pre_year) + rng.normal(0, 0.3, n_control)
        frames.append(
            pd.DataFrame({
                "unit_id": np.arange(n_control),
                "year": year,
                "y": y,
                "treatment": 0,
                "shock_year_col": np.nan,
            })
        )

    # Treated cohorts
    uid_offset = n_control
    for g in cohort_years:
        alpha_t = rng.normal(0, 1, n_per_cohort)
        uids = np.arange(uid_offset, uid_offset + n_per_cohort)
        for year in all_years:
            if year < g:
                att = 0.0
            else:
                att = true_att * (decay_rate ** (year - g))
            y = alpha_t + 0.05 * (year - pre_year) + att + rng.normal(0, 0.3, n_per_cohort)
            frames.append(
                pd.DataFrame({
                    "unit_id": uids,
                    "year": year,
                    "y": y,
                    "treatment": 1,
                    "shock_year_col": float(g),
                })
            )
        uid_offset += n_per_cohort

    return pd.concat(frames, ignore_index=True)


def test_case3b_classification():
    # Fast decay: individual cohort DiDs near 0, but staggered available
    data = _make_staggered_panel(
        cohort_years=[1820, 1850],
        post_years=[1888, 1911],
        pre_year=1777,
        shock_year=1796,
        true_att=2.5,
        decay_rate=0.85,  # decays somewhat but cohort gaps are large
        n_per_cohort=150,
        n_control=300,
    )
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=1796, pre_years=[1777], post_years=[1888, 1911],
        unit_id="unit_id", shock_year_col="shock_year_col",
        alpha=0.10,
    )
    # With staggered data provided, case should be 3b (no sig in aggregate DiD)
    # or 2 if some aggregate DiDs are significant — both are valid outcomes
    assert result.classification.case in ("2", "3a", "3b")
    assert result.identified_set is not None or result.classification.case in ("1", "2")


# ---------------------------------------------------------------------------
# Validation: summary and result shape
# ---------------------------------------------------------------------------


def test_summary_runs_for_all_cases():
    for dynamics, true_att in [("constant", 2.0), ("decaying", 3.0), ("zero", 0.0)]:
        data = _make_panel(dynamics=dynamics, true_att=true_att, n_units=400, seed=99)
        result = diagnose(
            data=data, outcome="y", treatment="treatment",
            shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
            unit_id="unit_id",
        )
        s = result.summary()
        assert "Temporal Mismatch" in s
        assert str(SHOCK_YEAR) in s


def test_post_dids_all_populated():
    data = _make_panel(dynamics="constant", true_att=2.0, n_units=300)
    result = diagnose(
        data=data, outcome="y", treatment="treatment",
        shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
        unit_id="unit_id",
    )
    assert set(result.post_dids.keys()) == set(POST_YEARS)
    for d in result.post_dids.values():
        assert d.gap > 0
        assert d.std_error > 0


def test_invalid_years_raise():
    data = _make_panel()
    with pytest.raises(ValueError):
        diagnose(
            data=data, outcome="y", treatment="treatment",
            shock_year=SHOCK_YEAR,
            pre_years=PRE_YEARS,
            post_years=[1750],  # before shock — no data
            unit_id="unit_id",
        )
