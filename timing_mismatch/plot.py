from __future__ import annotations

from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .diagnostics import DiagnosticsOutput, TemporalMismatchResult


def plot_case_diagram() -> plt.Figure:
    """
    Conceptual parallel-trends diagram illustrating the four temporal mismatch cases.

    Journal-style figure (minimal color, clean axes, serif-style fonts).
    Each panel shows stylised treatment and control group trajectories across
    pre-shock cross-sections, the unobserved shock year t*, and post-shock
    cross-sections.  Purely illustrative — no real data required.
    """
    # --- Global style parameters (journal-quality) ---
    rc = {
        "font.family":       "serif",
        "font.size":         10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.color":        "0.88",
        "grid.linewidth":    0.6,
        "grid.linestyle":    "-",
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "xtick.major.size":  3.5,
        "ytick.major.size":  3.5,
        "legend.frameon":    False,
        "legend.fontsize":   8.5,
        "figure.dpi":        150,
    }

    # Timeline
    PRE   = [1777, 1790]
    SHOCK = 1796
    POST  = [1820, 1860, 1900]
    ALL   = PRE + [SHOCK] + POST
    t_arr = np.array(ALL, dtype=float)
    obs_years = PRE + POST

    def baseline(t):
        return 0.018 * (np.asarray(t, float) - 1777)

    def eff_constant(t):
        return np.where(t >= SHOCK, 1.0, 0.0)

    def eff_decaying(t, rho=0.965):
        g = t - SHOCK
        return np.where(g >= 0, 1.0 * (rho ** g), 0.0)

    def eff_fast_decay(t, rho=0.82):
        g = t - SHOCK
        return np.where(g >= 0, 1.0 * (rho ** g), 0.0)

    def eff_late_cohort(t, cohort=1830, rho=0.90):
        g = t - cohort
        return np.where(g >= 0, 1.0 * (rho ** g), 0.0)

    # Monochrome palette
    BLACK  = "#111111"
    DGRAY  = "#555555"
    LGRAY  = "#aaaaaa"

    panel_labels = [r"\textbf{(a)}", r"\textbf{(b)}", r"\textbf{(c)}", r"\textbf{(d)}"]
    # fall back if usetex is off
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5),
                                 gridspec_kw=dict(hspace=0.42, wspace=0.32))

        # Shared helpers -------------------------------------------------------
        def _shock_line(ax, x=SHOCK, label=True):
            ax.axvline(x, color=DGRAY, ls=(0, (4, 3)), lw=0.9, zorder=2)
            if label:
                ylim = ax.get_ylim()
                ax.text(x + 2, ylim[0] + 0.92 * (ylim[1] - ylim[0]),
                        r"$t^{\ast}$", fontsize=9, color=DGRAY, va="top")

        def _obs_markers(ax, y_vals, style):
            """Mark observed cross-section years with filled/open markers."""
            for yr in obs_years:
                idx = ALL.index(yr)
                ax.plot(yr, y_vals[idx], **style)

        def _finish(ax, label, title, ylabel=True):
            ax.set_xlabel("Year", labelpad=3)
            if ylabel:
                ax.set_ylabel("Outcome", labelpad=3)
            ax.set_xticks(obs_years)
            ax.set_xticklabels(obs_years, rotation=30, ha="right", fontsize=8)
            ax.tick_params(labelsize=8.5)
            ax.set_title(title, fontsize=9.5, pad=6, loc="left")
            ax.text(-0.09, 1.04, label, transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom")

        # ------------------------------------------------------------ Case 1
        ax = axes[0, 0]
        y_t = baseline(t_arr) + eff_constant(t_arr) + 0.8
        y_c = baseline(t_arr)
        ax.plot(t_arr, y_t, "-",  color=BLACK, lw=1.6, label="Treated")
        ax.plot(t_arr, y_c, "--", color=DGRAY, lw=1.6, label="Control")
        _obs_markers(ax, y_t, dict(marker="o", ms=5, color=BLACK,  ls="none", zorder=4))
        _obs_markers(ax, y_c, dict(marker="s", ms=5, color=DGRAY,  ls="none", zorder=4,
                                   markerfacecolor="white", markeredgewidth=1.2))
        _shock_line(ax)
        ax.legend(loc="upper left")
        ax.set_ylim(ax.get_ylim()[0] - 0.05, ax.get_ylim()[1] + 0.12)
        _finish(ax, panel_labels[0],
                "Case 1 — Effect persists or grows\n"
                r"$\Rightarrow$ Standard DiD valid; report $\widehat{\mathrm{ATT}}(t_k)$")

        # ------------------------------------------------------------ Case 2
        ax = axes[0, 1]
        y_t2 = baseline(t_arr) + eff_decaying(t_arr) + 0.8
        y_c2 = baseline(t_arr)
        ax.plot(t_arr, y_t2, "-",  color=BLACK, lw=1.6, label="Treated")
        ax.plot(t_arr, y_c2, "--", color=DGRAY, lw=1.6, label="Control")
        _obs_markers(ax, y_t2, dict(marker="o", ms=5, color=BLACK, ls="none", zorder=4))
        _obs_markers(ax, y_c2, dict(marker="s", ms=5, color=DGRAY, ls="none", zorder=4,
                                    markerfacecolor="white", markeredgewidth=1.2))
        _shock_line(ax)
        ax.legend(loc="upper left")
        ax.set_ylim(ax.get_ylim()[0] - 0.05, ax.get_ylim()[1] + 0.12)
        _finish(ax, panel_labels[1],
                "Case 2 — Visible decay across post periods\n"
                r"$\Rightarrow$ Joint estimation of $\rho$ and $\mathrm{ATT}(t^{\ast})$",
                ylabel=False)

        # ----------------------------------------------------------- Case 3a
        ax = axes[1, 0]
        y_t3 = baseline(t_arr) + eff_fast_decay(t_arr) + 0.8
        y_c3 = baseline(t_arr)
        ax.plot(t_arr, y_t3, "-",  color=BLACK, lw=1.6, label="Treated")
        ax.plot(t_arr, y_c3, "--", color=DGRAY, lw=1.6, label="Control")
        _obs_markers(ax, y_t3, dict(marker="o", ms=5, color=BLACK, ls="none", zorder=4))
        _obs_markers(ax, y_c3, dict(marker="s", ms=5, color=DGRAY, ls="none", zorder=4,
                                    markerfacecolor="white", markeredgewidth=1.2))
        # Annotate the invisible gap at 1820
        idx_1820 = ALL.index(1820)
        gap = y_t3[idx_1820] - y_c3[idx_1820]
        ax.annotate("", xy=(1820, y_c3[idx_1820] + gap),
                    xytext=(1820, y_c3[idx_1820]),
                    arrowprops=dict(arrowstyle="<->", color=LGRAY, lw=0.9))
        ax.text(1822, y_c3[idx_1820] + gap / 2,
                r"$\approx 0$", fontsize=8, color=LGRAY, va="center")
        _shock_line(ax)
        ax.legend(loc="upper left")
        ax.set_ylim(ax.get_ylim()[0] - 0.05, ax.get_ylim()[1] + 0.12)
        _finish(ax, panel_labels[2],
                "Case 3a — Effect decayed before first observation\n"
                r"$\Rightarrow$ Sensitivity bounds on $\mathrm{ATT}(t^{\ast})$ over $\rho$")

        # ----------------------------------------------------------- Case 3b
        ax = axes[1, 1]
        y_te  = baseline(t_arr) + eff_fast_decay(t_arr) + 0.8
        y_tl  = baseline(t_arr) + eff_late_cohort(t_arr) + 1.1
        y_c3b = baseline(t_arr)
        ax.plot(t_arr, y_te,  "-",        color=BLACK, lw=1.6,
                label=r"Early cohort ($t^{\ast}$=1796)")
        ax.plot(t_arr, y_tl,  linestyle=(0, (6, 2)), color=BLACK, lw=1.6,
                label=r"Late cohort ($t^{\ast}$=1830)")
        ax.plot(t_arr, y_c3b, "--",       color=DGRAY, lw=1.6, label="Never treated")
        for yr in obs_years:
            idx = ALL.index(yr)
            ax.plot(yr, y_te[idx],  "o", ms=5, color=BLACK, zorder=4)
            ax.plot(yr, y_tl[idx],  "^", ms=5, color=BLACK, zorder=4)
            ax.plot(yr, y_c3b[idx], "s", ms=5, color=DGRAY,  zorder=4,
                    markerfacecolor="white", markeredgewidth=1.2)
        # Mark both shock years
        _shock_line(ax, x=SHOCK, label=False)
        _shock_line(ax, x=1830,  label=False)
        ylim = ax.get_ylim()
        mid  = ylim[0] + 0.88 * (ylim[1] - ylim[0])
        ax.text(SHOCK + 1, mid, r"$t^{\ast}_E$", fontsize=8.5, color=DGRAY)
        ax.text(1831,      mid, r"$t^{\ast}_L$", fontsize=8.5, color=DGRAY)
        ax.legend(loc="upper left", fontsize=7.8)
        ax.set_ylim(ylim[0] - 0.05, ylim[1] + 0.12)
        _finish(ax, panel_labels[3],
                "Case 3b — No aggregate effect; staggered timing available\n"
                r"$\Rightarrow$ Use cohort variation to identify $\rho$",
                ylabel=False)

        # Figure note
        fig.text(
            0.5, -0.02,
            r"Notes: Dots ($\bullet$) = observed cross-sections. "
            r"Dashed vertical lines mark policy shock years $t^{\ast}$ (unobserved in data). "
            r"All trajectories are stylised.",
            ha="center", fontsize=8, color=DGRAY,
            wrap=True,
        )

    return fig


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
