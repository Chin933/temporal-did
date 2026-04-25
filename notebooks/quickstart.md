# Quickstart Walkthrough

This walkthrough replicates a stylised historical scenario: an **1800 reform** with
county-level data available only at **1796** and **1820**.

## Install

```bash
git clone https://github.com/Chin933/temporal-did.git
cd temporal-did && pip install -e .
```

---

## 1. Simulate Panel Data

We generate 300 counties under a **decaying treatment effect** DGP:

$$\text{ATT}(t) = 1.0 \times 0.95^{\,t - 1800}$$

Only the 1796 and 1820 cross-sections are observed. $t^* = 1800$ is not.

```python
from timing_mismatch.monte_carlo import simulate_panel

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
# Shape: (600, 4)  —  300 units x 2 cross-sections
# Columns: unit_id, year, y, treatment
```

---

## 2. Run Diagnostics

```python
from timing_mismatch import timing_mismatch_diagnostics

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
```

Expected output:

```
Timing Mismatch Diagnostics
  Cross-sections : t1=1796, t2=1820
  Shock year     : t*=1800
  Position tau   : 0.167  (closer to t1)
  Mismatch sev.  : 0.556  (MODERATE)

  Strategy           Estimate       SE            95% CI  Notes
  -------------------------------------------------------------------------------------
  standard              ~0.36   ...   Estimates ATT(t2=1820); may under-estimate ATT(t*)
  ar1_adjusted          ~1.01   ...   ATT(t*) under AR(1) decay rho=0.95
  monotone_lb           ~0.36   ...   Lower bound on ATT(t*)
```

**Reading the output**:

| Number | Meaning |
|--------|---------|
| `tau = 0.167` | The shock at 1800 is 4/24 of the way from 1796 to 1820 — close to $t_1$ |
| `severity = 0.556` | Moderate. If the shock were at 1808 (midpoint), severity = 1.0 |
| `standard ≈ 0.36` | Estimates ATT(1820) = $1.0 \times 0.95^{20} \approx 0.36$ — large downward bias |
| `ar1_adjusted ≈ 1.01` | Correctly recovers ATT(1800) ≈ 1.0 (assuming the right $\rho$) |

---

## 3. Visualise

```python
from timing_mismatch import plot_diagnostics

fig = plot_diagnostics(result, true_att=1.0)
```

**Left panel** — Strategy comparison (95% CI). The dashed line is the true ATT = 1.0.

**Right panel** — Sensitivity of implied ATT($t^*$) to assumed $\rho$:
- Steep slope here because the gap $t_2 - t^* = 20$ years amplifies any deviation from $\rho = 1$
- The $\rho = 0.95$ point recovers the true ATT
- If the curve were flat near $\rho = 1$, the standard DiD would be robust regardless of dynamics

---

## 4. Monte Carlo Validation

500 independent simulations under the same DGP. True ATT$(t^*) = 1.0$.

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
    adjustment_rho=0.95,
    seed=0,
)
print(mc.attrs["summary"])
```

Expected summary:

```
              mean_bias    rmse  std_estimate
strategy
ar1_adjusted     ~0.00  ~0.16        ~0.04   <- near-zero bias, correctly specified
monotone_lb      ~-0.64  ~0.64        ~0.04   <- valid lower bound, not a point estimate
standard         ~-0.64  ~0.64        ~0.04   <- estimates ATT(t2), not ATT(t*)
```

```python
fig = plot_monte_carlo(mc)
```

**Takeaway**: When $\rho$ is correctly specified, `ar1_adjusted` has near-zero bias and RMSE driven only by sampling variance. The 64% downward bias of `standard` is purely from the timing mismatch — it would disappear if the data happened to include a 1800 cross-section.

---

## What Changes If You Have a Smaller Gap?

Try `post_year=1802` (2-year gap instead of 20):

```python
data_small = simulate_panel(post_year=1802, shock_year=1800, pre_year=1796, ...)
result_small = timing_mismatch_diagnostics(..., post_year=1802, shock_year=1800, ...)
print(result_small.mismatch_severity)  # severity drops dramatically
# 0.95^2 = 0.90  ->  standard DiD bias is only ~10%, not ~64%
```

The smaller the gap $t_2 - t^*$, the less timing mismatch matters — even under substantial decay.
