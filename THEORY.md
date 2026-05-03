# Theoretical Background

## Setup

Let $N$ units be observed at cross-sections $\mathcal{T} = \{t_1, \ldots, t_K\}$. A policy shock occurs at $t^{\ast} \notin \mathcal{T}$ — the shock year is never directly observed. Denote pre-shock years $\mathcal{T}^- = \{t \in \mathcal{T} : t < t^{\ast}\}$ and post-shock years $\mathcal{T}^+ = \{t \in \mathcal{T} : t > t^{\ast}\}$.

Unit $i$ receives binary treatment $D_i \in \{0, 1\}$. The **target parameter** is:

$$\text{ATT}(t^{\ast}) = \mathbb{E}\bigl[Y_{it^{\ast}}(1) - Y_{it^{\ast}}(0) \mid D_i = 1\bigr]$$

We never observe $Y_{it^{\ast}}$ — the shock year is not a data-collection year.

---

## Standard DiD and the Mismatch

**Assumption (Parallel Trends):** For all $s, t$,

$$\mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid D_i = 1] = \mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid D_i = 0]$$

Under parallel trends, a DiD using any $(t_{\text{pre}}, t_k)$ pair identifies $\text{ATT}(t_k)$:

$$\text{DiD}(t_{\text{pre}}, t_k) = \text{ATT}(t_k), \qquad t_k \in \mathcal{T}^+$$

This equals $\text{ATT}(t^{\ast})$ **only when treatment effects are constant over time**.

---

## Treatment Effect Dynamics: AR(1) Model

Suppose post-shock effects follow an AR(1) process:

$$\text{ATT}(t) = \text{ATT}(t^{\ast}) \cdot \rho^{\,t - t^{\ast}}, \qquad t \geq t^{\ast}$$

Then for any post-shock cross-section $t_k$:

$$\text{DiD}(t_{\text{pre}}, t_k) = \text{ATT}(t^{\ast}) \cdot \rho^{\,t_k - t^{\ast}}$$

| $\rho$ | Dynamics | Bias of standard DiD |
|--------|----------|----------------------|
| $< 1$ | Decaying | Negative — under-estimates ATT$(t^{\ast})$ |
| $= 1$ | Constant | Zero — unbiased |
| $> 1$ | Growing  | Positive — over-estimates ATT$(t^{\ast})$ |

---

## Case Classification

With multiple post-shock cross-sections, the pattern of $\{\text{DiD}(t_{\text{pre}}, t_k)\}_{t_k \in \mathcal{T}^+}$ determines what can be identified.

### Case 1 — Effect persists or grows ($\rho \geq 1$)

All post-shock DiDs are significant and stable or increasing over time. Standard DiD is valid. Report dynamic ATT$(t_k)$ at each cross-section.

### Case 2 — Effect decays, still visible ($\rho < 1$, effect detectable)

At least one post-shock DiD is significant, and the sequence is declining. Taking logarithms of the AR(1) model:

$$\log|\text{DiD}(t_{\text{pre}}, t_k)| = \underbrace{\log|\text{ATT}(t^{\ast})|}_{\text{intercept}} + (t_k - t^{\ast}) \cdot \underbrace{\log\rho}_{\text{slope}}$$

This is a linear model in event time $(t_k - t^{\ast})$. Fitting by WLS (weights $= 1/\text{SE}^2$) across all post-cross-sections jointly identifies both $\rho$ and ATT$(t^{\ast})$ — **without assuming $\rho$**.

**With staggered timing**: If different cohorts $g$ are treated at different times, define event time $\tau_{gk} = t_k - g$ for cohort $g$ observed at calendar time $t_k$. Under homogeneous ATT$(t^{\ast})$:

$$\log|\text{DiD}(g, t_k)| = \log|\text{ATT}(t^{\ast})| + \tau_{gk} \cdot \log\rho$$

The cross-cohort variation in $\tau_{gk}$ at the same calendar time identifies $\rho$ separately from ATT$(t^{\ast})$.

### Case 3a — Effect decayed before first post-period

All post-shock DiDs are statistically indistinguishable from zero. Under AR(1):

$$\text{ATT}(t^{\ast}) = \frac{\text{DiD}(t_{\text{pre}}, t_k)}{\rho^{t_k - t^{\ast}}}$$

Since DiD $\approx 0$, we cannot pin down ATT$(t^{\ast})$ without knowing $\rho$. The **identified set** for a given $\rho$ is:

$$\text{ATT}(t^{\ast}) \in \left[\frac{\text{CI}^{\text{lo}}_{t_k}}{\rho^{t_k - t^{\ast}}},\ \frac{\text{CI}^{\text{hi}}_{t_k}}{\rho^{t_k - t^{\ast}}}\right]$$

As $\rho \to 0$, the interval expands toward $\pm\infty$: rapid decay makes any ATT$(t^{\ast})$ consistent with the data. The tightest bounds come from the earliest $t_k \in \mathcal{T}^+$ (shortest gap).

### Case 3b — All post DiDs $\approx 0$, staggered timing available

A late-treated cohort $g$ may still show a detectable effect at calendar times where an early-treated cohort already shows zero. This differential visibility identifies $\rho$ via the staggered log-linear regression above. The Case 3a identified set is also reported as a fallback.

---

## Mismatch Severity (Two-Period Special Case)

When only a single pre-post pair $(t_1, t_2)$ is available, define the **temporal position**:

$$\tau = \frac{t^{\ast} - t_1}{t_2 - t_1} \in (0, 1)$$

The **mismatch severity**:

$$\text{severity}(\tau) = 4\tau(1-\tau) \in [0, 1]$$

equals 1 when the shock falls at the midpoint (worst case), and 0 when it coincides with a cross-section year.

**AR(1)-adjusted estimator** (point-identified given $\rho$):

$$\widehat{\text{ATT}}(t^{\ast}) = \frac{\text{DiD}(t_1, t_2)}{\rho^{t_2 - t^{\ast}}}$$

**Monotone lower bound**: If treatment effects are non-increasing after the shock ($\text{ATT}(t) \leq \text{ATT}(t^{\ast})$ for $t \geq t^{\ast}$), then DiD$(t_1, t_2) = \text{ATT}(t_2) \leq \text{ATT}(t^{\ast})$ is a sharp lower bound.

---

## Summary of Estimators

| Case | Data requirement | Estimator | Identifies |
|------|-----------------|-----------|------------|
| 1 | Multiple post cross-sections, stable DiDs | Dynamic DiD at each $t_k$ | ATT$(t_k)$ for each $t_k$ |
| 2 | $\geq 2$ post cross-sections or staggered timing | Log-linear WLS | ATT$(t^{\ast})$ and $\rho$ jointly |
| 3a | $\geq 1$ post cross-section | Sensitivity analysis over $\rho$ | Identified set for ATT$(t^{\ast})$ |
| 3b | Staggered timing | Staggered log-linear WLS + identified set | ATT$(t^{\ast})$ and $\rho$ (where cohort variation is sufficient) |
