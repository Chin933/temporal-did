import numpy as np
import pytest

from timing_mismatch import timing_mismatch_diagnostics
from timing_mismatch.monte_carlo import simulate_panel


def _make_data(seed: int = 0):
    return simulate_panel(
        n_units=100,
        pre_year=1796,
        shock_year=1800,
        post_year=1820,
        true_att=1.0,
        dynamics="constant",
        seed=seed,
    )


def test_output_keys():
    result = timing_mismatch_diagnostics(
        data=_make_data(),
        outcome="y",
        treatment="treatment",
        shock_year=1800,
        pre_year=1796,
        post_year=1820,
        unit_id="unit_id",
    )
    assert set(result.strategies.keys()) == {"standard", "ar1_adjusted", "monotone_lb"}
    assert 0.0 <= result.mismatch_severity <= 1.0
    assert 0.0 < result.tau < 1.0
    assert len(result.sensitivity) == 100


def test_midpoint_severity():
    # shock at midpoint -> tau=0.5, severity=1.0
    result = timing_mismatch_diagnostics(
        data=_make_data(),
        outcome="y",
        treatment="treatment",
        shock_year=1808,
        pre_year=1796,
        post_year=1820,
        unit_id="unit_id",
    )
    assert abs(result.tau - 0.5) < 1e-9
    assert abs(result.mismatch_severity - 1.0) < 1e-9


def test_ar1_identity_at_rho1():
    # rho=1 and gap=1 -> adjustment factor = 1 -> ar1_adjusted == standard
    result = timing_mismatch_diagnostics(
        data=_make_data(),
        outcome="y",
        treatment="treatment",
        shock_year=1819,
        pre_year=1796,
        post_year=1820,
        unit_id="unit_id",
        adjustment_rho=1.0,
    )
    assert abs(
        result.strategies["ar1_adjusted"].estimate
        - result.strategies["standard"].estimate
    ) < 1e-10


def test_invalid_year_order():
    with pytest.raises(ValueError, match="pre_year < shock_year < post_year"):
        timing_mismatch_diagnostics(
            data=_make_data(),
            outcome="y",
            treatment="treatment",
            shock_year=1790,
            pre_year=1796,
            post_year=1820,
            unit_id="unit_id",
        )


def test_sensitivity_at_rho1():
    result = timing_mismatch_diagnostics(
        data=_make_data(),
        outcome="y",
        treatment="treatment",
        shock_year=1800,
        pre_year=1796,
        post_year=1820,
        unit_id="unit_id",
        decay_rates=[0.8, 0.9, 1.0, 1.1, 1.2],
    )
    row = result.sensitivity[result.sensitivity["decay_rate"] == 1.0]
    assert len(row) == 1
    # At rho=1: implied_att = DiD / 1^gap = DiD = standard estimate
    assert abs(float(row["implied_att"].iloc[0]) - result.strategies["standard"].estimate) < 1e-10


def test_summary_output():
    result = timing_mismatch_diagnostics(
        data=_make_data(),
        outcome="y",
        treatment="treatment",
        shock_year=1800,
        pre_year=1796,
        post_year=1820,
        unit_id="unit_id",
    )
    s = result.summary()
    assert "Timing Mismatch Diagnostics" in s
    assert "standard" in s
    assert "ar1_adjusted" in s


def test_constant_dynamics_unbiased():
    # Under constant dynamics, standard DiD should be close to true ATT
    estimates = []
    for seed in range(30):
        data = simulate_panel(
            n_units=500, pre_year=1796, shock_year=1800, post_year=1820,
            true_att=1.0, dynamics="constant", seed=seed,
        )
        result = timing_mismatch_diagnostics(
            data=data, outcome="y", treatment="treatment",
            shock_year=1800, pre_year=1796, post_year=1820, unit_id="unit_id",
        )
        estimates.append(result.strategies["standard"].estimate)
    assert abs(np.mean(estimates) - 1.0) < 0.05
