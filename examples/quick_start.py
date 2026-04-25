"""
Quick start: 1800 Qing dynasty reform with decaying treatment effects.
Cross-sections available at 1796 and 1820 only.
"""

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script use
import matplotlib.pyplot as plt

from timing_mismatch import timing_mismatch_diagnostics, plot_diagnostics
from timing_mismatch.monte_carlo import simulate_panel, run_monte_carlo
from timing_mismatch.plot import plot_monte_carlo

# ── 1. Simulate data with known DGP ─────────────────────────────────────────
data = simulate_panel(
    n_units=300,
    pre_year=1796,
    shock_year=1800,
    post_year=1820,
    true_att=1.0,
    dynamics="decaying",
    decay_rate=0.95,
    seed=42,
)
print(f"Panel shape: {data.shape}")
print(data.head(), "\n")

# ── 2. Run diagnostics ───────────────────────────────────────────────────────
result = timing_mismatch_diagnostics(
    data=data,
    outcome="y",
    treatment="treatment",
    shock_year=1800,
    pre_year=1796,
    post_year=1820,
    unit_id="unit_id",
    adjustment_rho=0.95,
)
print(result.summary())

# ── 3. Plot diagnostics ──────────────────────────────────────────────────────
fig = plot_diagnostics(result, true_att=1.0)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
print("\nSaved: diagnostics.png")

# ── 4. Monte Carlo: 500 simulations under decaying DGP ──────────────────────
mc = run_monte_carlo(
    n_simulations=500,
    shock_year=1800,
    pre_year=1796,
    post_year=1820,
    true_att=1.0,
    dynamics="decaying",
    decay_rate=0.95,
    adjustment_rho=0.95,
    seed=0,
)
print("\nMonte Carlo summary:")
print(mc.attrs["summary"].to_string())

fig_mc = plot_monte_carlo(mc)
fig_mc.savefig("monte_carlo.png", dpi=150, bbox_inches="tight")
print("\nSaved: monte_carlo.png")
