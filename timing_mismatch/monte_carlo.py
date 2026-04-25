from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd

from .diagnostics import timing_mismatch_diagnostics


def simulate_panel(
    n_units: int = 200,
    treatment_share: float = 0.5,
    pre_year: int = 1796,
    shock_year: int = 1800,
    post_year: int = 1820,
    true_att: float = 1.0,
    dynamics: Literal["constant", "decaying", "growing"] = "constant",
    decay_rate: float = 0.95,
    sigma: float = 0.5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Simulate historical panel data with two cross-sections and a known DGP.

    Parameters
    ----------
    n_units : int
        Number of panel units.
    treatment_share : float
        Share of units assigned to treatment.
    pre_year, shock_year, post_year : int
        Cross-section years and shock year (only pre and post are observed).
    true_att : float
        True ATT at t* under the DGP.
    dynamics : {"constant", "decaying", "growing"}
        How ATT evolves after t*.
    decay_rate : float
        Per-year factor for the decaying/growing dynamics.
    sigma : float
        Noise standard deviation.
    seed : int, optional

    Returns
    -------
    DataFrame with columns: unit_id, year, y, treatment.
    Only pre_year and post_year rows are included (t* is not observed).
    """
    rng = np.random.default_rng(seed)
    n_treated = int(n_units * treatment_share)
    n_control = n_units - n_treated

    unit_ids = np.arange(n_units)
    d = np.array([1] * n_treated + [0] * n_control)
    alpha = rng.normal(0, 1, n_units)

    rows: List[dict] = []
    for year in (pre_year, post_year):
        time_effect = 0.1 * (year - pre_year)

        if year < shock_year:
            att = 0.0
        else:
            t_since = year - shock_year
            if dynamics == "constant":
                att = true_att
            elif dynamics == "decaying":
                att = true_att * (decay_rate**t_since)
            else:  # growing
                att = true_att * ((2 - decay_rate) ** t_since)

        y = alpha + time_effect + att * d + rng.normal(0, sigma, n_units)
        rows.extend(
            {"unit_id": int(uid), "year": year, "y": float(yi), "treatment": int(di)}
            for uid, yi, di in zip(unit_ids, y, d)
        )

    return pd.DataFrame(rows)


def run_monte_carlo(
    n_simulations: int = 500,
    n_units: int = 200,
    pre_year: int = 1796,
    shock_year: int = 1800,
    post_year: int = 1820,
    true_att: float = 1.0,
    dynamics: Literal["constant", "decaying", "growing"] = "constant",
    decay_rate: float = 0.95,
    adjustment_rho: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run Monte Carlo simulations to evaluate bias and RMSE of alignment strategies.

    Parameters
    ----------
    n_simulations : int
        Number of simulation draws.
    n_units : int
        Panel units per simulation.
    pre_year, shock_year, post_year : int
        Temporal structure.
    true_att : float
        Ground-truth ATT at t*.
    dynamics : {"constant", "decaying", "growing"}
        True DGP for treatment effect dynamics.
    decay_rate : float
        Per-year decay/growth factor under the DGP.
    adjustment_rho : float
        Assumed rho for the ar1_adjusted strategy.
    seed : int
        Master random seed.

    Returns
    -------
    DataFrame with columns: simulation, strategy, estimate, true_att, bias.
    A ``summary`` attribute (DataFrame) is attached with per-strategy bias/RMSE.
    """
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10**6, n_simulations)

    records: List[dict] = []
    for i, s in enumerate(seeds):
        data = simulate_panel(
            n_units=n_units,
            pre_year=pre_year,
            shock_year=shock_year,
            post_year=post_year,
            true_att=true_att,
            dynamics=dynamics,
            decay_rate=decay_rate,
            seed=int(s),
        )
        result = timing_mismatch_diagnostics(
            data=data,
            outcome="y",
            treatment="treatment",
            shock_year=shock_year,
            pre_year=pre_year,
            post_year=post_year,
            unit_id="unit_id",
            adjustment_rho=adjustment_rho,
        )
        for sname, sres in result.strategies.items():
            records.append(
                {
                    "simulation": i,
                    "strategy": sname,
                    "estimate": sres.estimate,
                    "true_att": true_att,
                    "bias": sres.estimate - true_att,
                }
            )

    df = pd.DataFrame(records)
    summary = (
        df.groupby("strategy")
        .agg(
            mean_bias=("bias", "mean"),
            rmse=("bias", lambda x: float(np.sqrt((x**2).mean()))),
            std_estimate=("estimate", "std"),
        )
        .round(4)
    )
    df.attrs["summary"] = summary
    return df
