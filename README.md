# timing-mismatch

A Python toolkit for diagnosing and correcting temporal misalignment in historical difference-in-differences research.

[![Tests](https://github.com/Chin933/qing_tax/actions/workflows/tests.yml/badge.svg)](https://github.com/Chin933/qing_tax/actions/workflows/tests.yml)

## The Problem

In historical empirical research, a policy shock at year $t^*$ often falls between available cross-sections $t_1 < t^* < t_2$. The naive approach — treating $t_2$ as the post-period — estimates the treatment effect at $t_2$, not at the shock year:

$$\text{DiD}(t_1, t_2) = \hat{\text{ATT}}(t_2) \neq \text{ATT}(t^*)$$

**Example:** You study an 1800 policy reform, but only have prefectural records for 1796 and 1820. Depending on whether treatment effects decay, persist, or grow over the 20-year window, the mismatch between your estimate and the true effect at $t^*$ can be substantial.

The bias depends on:
- **Temporal position** — how close is $t^*$ to each cross-section?
- **Treatment effect dynamics** — do effects decay, persist, or accumulate?
- **Pre-trend shape** — linear or non-linear?

## What This Package Does

Given your cross-section years and shock year, `timing-mismatch` provides:

| Output | Description |
|--------|-------------|
| **Mismatch severity** | $4\tau(1-\tau)$ where $\tau = (t^* - t_1)/(t_2 - t_1)$; equals 1 at the midpoint |
| **Three alignment estimates** | Under different assumptions about treatment dynamics |
| **Sensitivity analysis** | Implied $\text{ATT}(t^*)$ as a function of assumed decay rate $\rho$ |
| **Monte Carlo** | Bias and RMSE under known DGPs |

## Installation

```bash
git clone https://github.com/Chin933/qing_tax.git
cd qing_tax
pip install -e .
```

## Quick Start

```python
import pandas as pd
from timing_mismatch import timing_mismatch_diagnostics, plot_diagnostics

# df: panel DataFrame with columns [county_id, year, outcome, treatment]
result = timing_mismatch_diagnostics(
    data=df,
    outcome="grain_tax",
    treatment="reform",
    shock_year=1800,       # t*: when the reform happened
    pre_year=1796,         # t1: earliest available cross-section
    post_year=1820,        # t2: post-shock cross-section
    unit_id="county_id",
)

print(result.summary())
# Timing Mismatch Diagnostics
#   Cross-sections : t1=1796, t2=1820
#   Shock year     : t*=1800
#   Position tau   : 0.167  (closer to t1)
#   Mismatch sev.  : 0.556  (MODERATE)
#   ...

fig = plot_diagnostics(result)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
```

## Alignment Strategies

### `standard`
Plain DiD using $(t_1, t_2)$. Estimates $\text{ATT}(t_2)$, not $\text{ATT}(t^*)$.

### `ar1_adjusted`
Adjusts for assumed AR(1) decay:

$$\widehat{\text{ATT}}(t^*) = \frac{\text{DiD}(t_1, t_2)}{\rho^{t_2 - t^*}}$$

Use the sensitivity plot to check robustness across values of $\rho$.

### `monotone_lb`
Under non-increasing treatment effects ($\text{ATT}(t) \leq \text{ATT}(t^*)$ for $t > t^*$), the standard DiD is a **lower bound** on $\text{ATT}(t^*)$.

## Sensitivity Analysis

The sensitivity plot shows $\widehat{\text{ATT}}(t^*)$ as a function of assumed $\rho$:
- $\rho = 1$: no dynamics — standard DiD is unbiased
- $\rho < 1$: decaying effects — standard DiD under-estimates
- $\rho > 1$: growing effects — standard DiD over-estimates

A flat curve near $\rho = 1$ means your estimate is robust to reasonable assumptions about dynamics.

## Monte Carlo Validation

```python
from timing_mismatch.monte_carlo import run_monte_carlo
from timing_mismatch.plot import plot_monte_carlo

mc = run_monte_carlo(
    n_simulations=500,
    shock_year=1800,
    pre_year=1796,
    post_year=1820,
    true_att=1.0,
    dynamics="decaying",
    decay_rate=0.95,
    adjustment_rho=0.95,  # correctly specified
)
print(mc.attrs["summary"])
#               mean_bias    rmse  std_estimate
# strategy
# standard        -0.0923  0.1012        0.0437
# ar1_adjusted     0.0011  0.0452        0.0492
# monotone_lb     -0.0923  0.1012        0.0437

fig = plot_monte_carlo(mc)
```

## Theoretical Background

The key parameter is the **temporal position**:

$$\tau = \frac{t^* - t_1}{t_2 - t_1} \in (0, 1)$$

**Mismatch severity** $= 4\tau(1-\tau)$ is maximized at $\tau = 0.5$ (shock exactly midway) and zero when the shock coincides with a cross-section year.

Under the AR(1) model $\text{ATT}(t) = \theta \cdot \rho^{t - t^*}$:

| Condition | Consequence |
|-----------|-------------|
| $\rho = 1$ (constant) | $\text{DiD}(t_1, t_2) = \text{ATT}(t^*)$; no bias |
| $\rho < 1$ (decaying) | $\text{DiD}(t_1, t_2) < \text{ATT}(t^*)$; standard DiD under-estimates |
| $\rho > 1$ (growing) | $\text{DiD}(t_1, t_2) > \text{ATT}(t^*)$; standard DiD over-estimates |

Under monotone non-increasing effects, $\text{DiD}(t_1, t_2)$ is a valid lower bound on $\text{ATT}(t^*)$ regardless of the exact decay process.

## Citation

```bibtex
@misc{zhou2026timing,
  title  = {Timing Mismatch in Historical Difference-in-Differences},
  author = {Zhou, Qinnan},
  year   = {2026},
  url    = {https://github.com/Chin933/qing_tax}
}
```

## License

MIT
