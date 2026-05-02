# Quickstart Walkthrough

This walkthrough uses a stylised historical scenario: a **1796 reform** with
county-level records available at **1777, 1820, 1888, and 1911** — the shock year
itself is never observed.

## Install

```bash
git clone https://github.com/Chin933/temporal-did.git
cd temporal-did && pip install -e .
```

---

## The Problem at a Glance

```python
from timing_mismatch import plot_case_diagram

fig = plot_case_diagram()
```

The diagram shows the four possible settings depending on what the post-shock
DiDs reveal. The toolkit detects which case applies automatically.

---

## 1. Simulate Multi-Period Panel Data

We generate 400 counties with a **slow-decay DGP**: ATT declines at $\rho = 0.97$
per year after 1796, so the effect is still visible in 1820 and 1888 but small by 1911.

```python
import numpy as np
import pandas as pd

CROSS_SECTIONS = [1777, 1820, 1888, 1911]
SHOCK_YEAR = 1796

rng = np.random.default_rng(42)
n_units, true_att, rho = 400, 2.0, 0.97
d = np.array([1] * 200 + [0] * 200)
alpha = rng.normal(0, 1, n_units)

frames = []
for year in CROSS_SECTIONS:
    gap = max(year - SHOCK_YEAR, 0)
    att = true_att * (rho ** gap) if year >= SHOCK_YEAR else 0.0
    y = alpha + 0.02 * (year - 1777) + att * d + rng.normal(0, 0.4, n_units)
    frames.append(pd.DataFrame({"unit_id": np.arange(n_units),
                                "year": year, "y": y, "treatment": d}))
data = pd.concat(frames, ignore_index=True)
# Shape: (1600, 4)  —  400 units × 4 cross-sections
```

---

## 2. Run Diagnostics

```python
from timing_mismatch import diagnose

result = diagnose(
    data=data,
    outcome="y",
    treatment="treatment",
    shock_year=1796,
    pre_years=[1777],
    post_years=[1820, 1888, 1911],
    unit_id="unit_id",
)
print(result.summary())
```

Expected output (Case 2 — decaying, still visible):

```
Temporal Mismatch Diagnostics
  Shock year   : t* = 1796
  Pre periods  : [1777]
  Post periods : [1820, 1888, 1911]
  Case         : 2  —  Effect decays but visible in post periods — jointly estimate rho and ATT(t*)
  Staggered    : no
  Significant post-DiDs : 2 / 3  (alpha=0.1)

  Strategy               Estimate       SE            95% CI  Notes
  ------------------------------------------------------------------------------------------
  joint_att_star           ~2.00     ...    ATT(t*) via joint log-linear; rho_hat≈0.97
  joint_rho                ~0.97     ...    Estimated per-year decay rate
```

**Reading the output**:

| Field | Meaning |
|-------|---------|
| `Case 2` | Effect is decaying and still detectable — joint estimation possible |
| `joint_att_star ≈ 2.00` | Recovered ATT$(t^{\ast}=1796)$; close to the true value of 2.0 |
| `joint_rho ≈ 0.97` | Estimated per-year decay rate; close to the true $\rho = 0.97$ |

---

## 3. What Each Case Looks Like

### Case 1 — constant effects

```python
# constant DGP: rho = 1
att_case1 = np.where(data["year"] >= SHOCK_YEAR, true_att, 0.0)
# result.classification.case == "1"
# strategies: att_1820, att_1888, att_1911 — all close to 2.0
```

The toolkit reports dynamic ATT$(t_k)$ at each post-shock cross-section.
All three should be near the true ATT, with no time trend.

### Case 2 — slow decay (this example)

Joint WLS on $\log|\text{DiD}_k| = \log|\text{ATT}(t^{\ast})| + (t_k - t^{\ast})\log\rho$
recovers both the shock-year effect and the decay rate simultaneously.

### Case 3a — fast decay, nothing visible

```python
result_3a = diagnose(
    data=data_zero_effect,          # all post DiDs ~ 0
    outcome="y", treatment="treatment",
    shock_year=1796, pre_years=[1777], post_years=[1820, 1888, 1911],
    unit_id="unit_id", alpha=0.001,
)
print(result_3a.identified_set.head())
#      rho  point_att_star  ci_lower  ci_upper  min_detectable
# ...
```

The toolkit returns an **identified set**: for each assumed $\rho$, the 95% CI
for ATT$(t^{\ast})$ implied by the near-zero DiD. As $\rho \to 0$ the bound widens —
a very fast decay makes any true ATT consistent with the data.

---

## 4. Visualise

```python
from timing_mismatch import plot_temporal_mismatch

fig = plot_temporal_mismatch(result, true_att=2.0)
fig.savefig("diagnostics.png", dpi=150, bbox_inches="tight")
```

**Left panel** — Post-shock DiD estimates over time, with 95% CIs.
Markers show only the observed cross-section years; the vertical dashed line marks $t^{\ast}$.

**Right panel** (Cases 1/2) — Estimated ATT$(t^{\ast})$ and $\rho$ with confidence intervals.
**Right panel** (Cases 3a/3b) — Identified-set bounds as a function of assumed $\rho$.

---

## 5. Monte Carlo Validation

```python
from timing_mismatch.monte_carlo import run_monte_carlo
from timing_mismatch.plot import plot_monte_carlo

mc = run_monte_carlo(
    n_simulations=500,
    shock_year=1800, pre_year=1796, post_year=1820,
    true_att=1.0, dynamics="decaying", decay_rate=0.95,
    adjustment_rho=0.95, seed=0,
)
print(mc.attrs["summary"])
```

Expected summary:

```
              mean_bias    rmse  std_estimate
strategy
ar1_adjusted     ~0.00   ~0.16        ~0.04   ← near-zero bias when rho is correct
monotone_lb     ~-0.64   ~0.64        ~0.04   ← valid lower bound, not a point estimate
standard        ~-0.64   ~0.64        ~0.04   ← estimates ATT(t2), not ATT(t*)
```

The 64% downward bias of `standard` is purely from timing mismatch — it
disappears when $\rho = 1$ (constant effects) or when the data happens to include
a cross-section at $t^{\ast}$.

---

## What If I Only Have Two Cross-Sections?

The legacy `timing_mismatch_diagnostics` handles the classical two-period case
when $\rho$ must be assumed rather than estimated:

```python
from timing_mismatch import timing_mismatch_diagnostics

result_legacy = timing_mismatch_diagnostics(
    data=data_two_period,
    outcome="y", treatment="treatment",
    shock_year=1800, pre_year=1796, post_year=1820,
    unit_id="unit_id",
    adjustment_rho=0.95,   # must be assumed
)
print(result_legacy.summary())
```

Use `diagnose()` whenever you have $\geq 2$ post-shock cross-sections —
it estimates $\rho$ from the data instead of requiring an assumption.
