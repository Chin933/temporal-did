from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import DiagnosticsOutput


def plot_sensitivity(
    result: DiagnosticsOutput,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Sensitivity of implied ATT(t*) to assumed per-year decay rate."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    sens = result.sensitivity
    ax.plot(sens["decay_rate"], sens["implied_att"], "b-", lw=2, label="Implied ATT(t*)")
    ax.axvline(1.0, color="gray", ls="--", alpha=0.6, label="No dynamics (rho=1)")
    ax.axhline(
        result.strategies["standard"].estimate,
        color="r",
        ls=":",
        alpha=0.7,
        label="Standard DiD",
    )
    ax.set_xlabel("Decay rate rho  (rho < 1: decaying, rho > 1: growing)")
    ax.set_ylabel("Implied ATT at shock year")
    ax.set_title(
        f"Sensitivity to Treatment Effect Dynamics\n"
        f"gap = {result.post_year} - {result.shock_year} = "
        f"{result.post_year - result.shock_year} yrs"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_strategy_comparison(
    result: DiagnosticsOutput,
    ax: Optional[plt.Axes] = None,
    true_att: Optional[float] = None,
) -> plt.Axes:
    """Horizontal bar chart comparing estimates across alignment strategies."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    strategies = list(result.strategies.values())
    names = [s.name for s in strategies]
    ests = [s.estimate for s in strategies]
    errs = [1.96 * s.std_error for s in strategies]

    y_pos = np.arange(len(strategies))
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]
    ax.barh(y_pos, ests, xerr=errs, align="center", alpha=0.7, capsize=5, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)

    if true_att is not None:
        ax.axvline(true_att, color="k", ls="--", lw=1.5, label=f"True ATT = {true_att}")
        ax.legend(fontsize=9)

    ax.set_xlabel("Estimate  (bars show 95% CI)")
    ax.set_title(
        f"Alignment Strategy Comparison\n"
        f"severity={result.mismatch_severity:.3f}, tau={result.tau:.3f}"
    )
    ax.grid(axis="x", alpha=0.3)
    return ax


def plot_diagnostics(
    result: DiagnosticsOutput,
    true_att: Optional[float] = None,
) -> plt.Figure:
    """Combined figure: strategy comparison + sensitivity analysis."""
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    plot_strategy_comparison(result, ax=fig.add_subplot(gs[0]), true_att=true_att)
    plot_sensitivity(result, ax=fig.add_subplot(gs[1]))
    fig.suptitle(
        f"Timing Mismatch Diagnostics  |  "
        f"t1={result.pre_year}, t*={result.shock_year}, t2={result.post_year}",
        fontsize=13,
    )
    return fig


def plot_monte_carlo(mc_results: pd.DataFrame) -> plt.Figure:
    """Bias distribution and RMSE / mean-bias comparison from Monte Carlo."""
    strategies = mc_results["strategy"].unique()
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for strat, col in zip(strategies, colors):
        bias = mc_results.loc[mc_results["strategy"] == strat, "bias"]
        ax.hist(bias, bins=40, alpha=0.55, density=True, color=col, label=strat)
    ax.axvline(0, color="k", ls="--", lw=1.5)
    ax.set_xlabel("Bias  (estimate - true ATT)")
    ax.set_ylabel("Density")
    ax.set_title("Bias Distribution")
    ax.legend(fontsize=9)

    ax = axes[1]
    summary = (
        mc_results.groupby("strategy")
        .agg(
            rmse=("bias", lambda x: float(np.sqrt((x**2).mean()))),
            mean_abs_bias=("bias", lambda x: float(x.abs().mean())),
        )
        .reset_index()
    )
    x = np.arange(len(summary))
    w = 0.35
    ax.bar(x - w / 2, summary["rmse"], w, label="RMSE", alpha=0.8, color="#4878CF")
    ax.bar(
        x + w / 2, summary["mean_abs_bias"], w, label="|Mean Bias|", alpha=0.8, color="#D65F5F"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(summary["strategy"], rotation=12, ha="right")
    ax.set_ylabel("Value")
    ax.set_title("RMSE and |Mean Bias| by Strategy")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig
