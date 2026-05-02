from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import DiagnosticsOutput, TemporalMismatchResult


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


def plot_post_dids(
    result: TemporalMismatchResult,
    ax: Optional[plt.Axes] = None,
    true_att: Optional[float] = None,
) -> plt.Axes:
    """Dynamic ATT(tk) for each post-shock cross-section with 95% CIs."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    years = sorted(result.post_dids.keys())
    ests = [result.post_dids[y].estimate for y in years]
    errs = [1.96 * result.post_dids[y].std_error for y in years]

    ax.errorbar(years, ests, yerr=errs, fmt="o-", capsize=5, lw=1.8,
                color="#4878CF", label="DiD estimate")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.axvline(result.shock_year, color="k", ls=":", lw=1.5,
               label=f"Shock t*={result.shock_year}")

    if true_att is not None:
        ax.axhline(true_att, color="r", ls="--", lw=1.5, alpha=0.7,
                   label=f"True ATT={true_att}")

    ax.set_xlabel("Year")
    ax.set_ylabel("DiD estimate")
    ax.set_title(
        f"Post-Shock DiD Estimates  |  Case {result.classification.case}\n"
        f"{result.classification.n_significant}/{result.classification.n_post} "
        f"significant at alpha={result.classification.alpha}"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_identified_set(
    result: TemporalMismatchResult,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Case 3a: ATT(t*) identified-set bounds as a function of assumed rho."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if result.identified_set is None:
        ax.text(0.5, 0.5, "No identified set (not Case 3a/3b)",
                ha="center", va="center", transform=ax.transAxes)
        return ax

    df = result.identified_set
    ax.fill_between(df["rho"], df["ci_lower"], df["ci_upper"],
                    alpha=0.25, color="#D65F5F", label="95% CI band")
    ax.plot(df["rho"], df["point_att_star"], color="#D65F5F", lw=2,
            label="Point ATT(t*) = DiD / rho^gap")
    ax.plot(df["rho"], df["min_detectable"], color="#6ACC65", lw=1.5,
            ls="--", label="Min. detectable ATT(t*)")
    ax.axhline(0, color="gray", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("Assumed decay rate rho")
    ax.set_ylabel("Implied ATT(t*)")
    ax.set_title(
        f"Identified Set  |  Case {result.classification.case}\n"
        f"gap = {df['gap_years'].iloc[0]} yrs  "
        f"(post_year={int(df['post_year'].iloc[0])}, t*={result.shock_year})"
    )
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    return ax


def plot_temporal_mismatch(
    result: TemporalMismatchResult,
    true_att: Optional[float] = None,
) -> plt.Figure:
    """
    Combined diagnostic figure for TemporalMismatchResult.

    - Left : post-shock DiD estimates over time
    - Right: case-specific panel (strategy comparison for Cases 1/2,
             identified-set bounds for Cases 3a/3b)
    """
    fig = plt.figure(figsize=(13, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    plot_post_dids(result, ax=fig.add_subplot(gs[0]), true_att=true_att)

    case = result.classification.case
    ax_right = fig.add_subplot(gs[1])

    if case in ("3a", "3b"):
        plot_identified_set(result, ax=ax_right)
    else:
        # Strategy comparison bar chart
        strategies = list(result.strategies.values())
        if strategies:
            names = [s.name for s in strategies]
            ests = [s.estimate for s in strategies]
            errs = [1.96 * s.std_error for s in strategies]
            y_pos = np.arange(len(strategies))
            colors = ["#4878CF", "#6ACC65", "#D65F5F", "#E78AC3"]
            ax_right.barh(y_pos, ests, xerr=errs, align="center",
                          alpha=0.75, capsize=5,
                          color=colors[: len(strategies)])
            if true_att is not None:
                ax_right.axvline(true_att, color="k", ls="--", lw=1.5,
                                 label=f"True ATT={true_att}")
                ax_right.legend(fontsize=9)
            ax_right.set_yticks(y_pos)
            ax_right.set_yticklabels(names)
            ax_right.set_xlabel("Estimate  (bars show 95% CI)")
            ax_right.set_title(f"Strategy Estimates  |  Case {case}")
            ax_right.grid(axis="x", alpha=0.3)

    fig.suptitle(
        f"Temporal Mismatch Diagnostics  |  "
        f"t*={result.shock_year}  "
        f"pre={result.pre_years}  post={result.post_years}",
        fontsize=12,
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
