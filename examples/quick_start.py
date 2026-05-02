"""
Quick start: 1796 reform, cross-sections at 1777 / 1820 / 1888 / 1911.

Demonstrates the automatic case classification in diagnose() across three
simulated DGPs:
  - Constant effects  → Case 1
  - Slow decay        → Case 2
  - Fast decay        → Case 3a
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from timing_mismatch import diagnose, plot_temporal_mismatch, plot_case_diagram
from timing_mismatch.monte_carlo import run_monte_carlo
from timing_mismatch.plot import plot_monte_carlo

CROSS_SECTIONS = [1777, 1820, 1888, 1911]
SHOCK_YEAR = 1796
PRE_YEARS = [1777]
POST_YEARS = [1820, 1888, 1911]


# ── helpers ──────────────────────────────────────────────────────────────────

def make_panel(dynamics, decay_rate=0.97, true_att=2.0, n_units=400, seed=0):
    """Simulate a multi-period panel with the given DGP."""
    rng = np.random.default_rng(seed)
    n_treated = n_units // 2
    n_control = n_units - n_treated
    unit_ids = np.arange(n_units)
    d = np.array([1] * n_treated + [0] * n_control)
    alpha = rng.normal(0, 1, n_units)

    frames = []
    for year in CROSS_SECTIONS:
        time_effect = 0.02 * (year - CROSS_SECTIONS[0])
        if year < SHOCK_YEAR:
            att = 0.0
        else:
            gap = year - SHOCK_YEAR
            if dynamics == "constant":
                att = true_att
            elif dynamics == "decaying":
                att = true_att * (decay_rate ** gap)
            else:  # zero
                att = 0.0
        y = alpha + time_effect + att * d + rng.normal(0, 0.4, n_units)
        frames.append(
            pd.DataFrame({"unit_id": unit_ids, "year": year, "y": y, "treatment": d})
        )
    return pd.concat(frames, ignore_index=True)


# ── 1. Conceptual diagram ─────────────────────────────────────────────────────

fig0 = plot_case_diagram()
fig0.savefig("case_diagram.png", dpi=150, bbox_inches="tight")
print("Saved: case_diagram.png\n")


# ── 2. Case 1: constant effects ───────────────────────────────────────────────

data1 = make_panel("constant", true_att=2.0)
r1 = diagnose(
    data=data1, outcome="y", treatment="treatment",
    shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
    unit_id="unit_id",
)
print("=== Case 1: constant effects ===")
print(r1.summary(), "\n")
plot_temporal_mismatch(r1, true_att=2.0).savefig("case1.png", dpi=150, bbox_inches="tight")


# ── 3. Case 2: slow decay (ρ ≈ 0.97), effect still visible ───────────────────

data2 = make_panel("decaying", decay_rate=0.97, true_att=3.0)
r2 = diagnose(
    data=data2, outcome="y", treatment="treatment",
    shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
    unit_id="unit_id",
)
print("=== Case 2: decaying effects ===")
print(r2.summary(), "\n")
print(f"  Estimated rho = {r2.estimated_rho:.3f}  (true = 0.97)")
plot_temporal_mismatch(r2, true_att=3.0).savefig("case2.png", dpi=150, bbox_inches="tight")


# ── 4. Case 3a: fast decay, nothing visible ───────────────────────────────────

data3 = make_panel("zero", true_att=0.0)
r3 = diagnose(
    data=data3, outcome="y", treatment="treatment",
    shock_year=SHOCK_YEAR, pre_years=PRE_YEARS, post_years=POST_YEARS,
    unit_id="unit_id", alpha=0.001,
)
print("=== Case 3a: no visible effect ===")
print(r3.summary(), "\n")
plot_temporal_mismatch(r3).savefig("case3a.png", dpi=150, bbox_inches="tight")


# ── 5. Monte Carlo (two-period legacy API) ────────────────────────────────────

mc = run_monte_carlo(
    n_simulations=500,
    shock_year=1800, pre_year=1796, post_year=1820,
    true_att=1.0, dynamics="decaying", decay_rate=0.95,
    adjustment_rho=0.95, seed=0,
)
print("=== Monte Carlo summary ===")
print(mc.attrs["summary"].to_string(), "\n")
plot_monte_carlo(mc).savefig("monte_carlo.png", dpi=150, bbox_inches="tight")
print("Saved: case1.png, case2.png, case3a.png, monte_carlo.png")
