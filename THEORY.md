# Theoretical Background

## Setup

Let $N$ units be observed at two cross-sections, $t_1$ (pre-shock) and $t_2$ (post-shock). A policy shock occurs at $t^* \in (t_1, t_2)$ — strictly between the two observation points. Unit $i$ receives treatment $D_i \in \{0, 1\}$.

**Target parameter**:

$$\text{ATT}(t^*) = \mathbb{E}\bigl[Y_{it^*}(1) - Y_{it^*}(0) \mid D_i = 1\bigr]$$

We never observe $Y_{it^*}$ — the problem is that $t^*$ is not a data-collection year.

---

## Standard DiD and the Mismatch

**Assumption (Parallel Trends)**: $\mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid D_i = 1] = \mathbb{E}[Y_{it}(0) - Y_{is}(0) \mid D_i = 0]$ for all $s, t$.

Under parallel trends, standard DiD using $(t_1, t_2)$ identifies:

$$\text{DiD}(t_1, t_2) = \mathbb{E}[Y_{it_2}(1) - Y_{it_2}(0) \mid D_i = 1] = \text{ATT}(t_2)$$

This equals $\text{ATT}(t^*)$ only when treatment effects are **constant over time**.

---

## Mismatch Severity

Define the **temporal position** of the shock:

$$\tau = \frac{t^* - t_1}{t_2 - t_1} \;\in\; (0, 1)$$

The **mismatch severity** is:

$$\text{severity}(\tau) = 4\tau(1-\tau) \;\in\; [0, 1]$$

| Value | Condition |
|-------|-----------|
| **1** | $t^* = (t_1 + t_2)/2$ — shock exactly at the midpoint (worst case) |
| **0** | $\tau \to 0$ or $\tau \to 1$ — shock coincides with a cross-section year |

Severity captures how exposed the estimate is to treatment effect dynamics. At severity 1, even small deviations from constant effects produce large biases.

---

## Bias under AR(1) Dynamics

Suppose treatment effects follow an AR(1) process after the shock:

$$\text{ATT}(t) = \text{ATT}(t^*) \cdot \rho^{\,t - t^*}, \qquad t \geq t^*$$

Then:

$$\text{DiD}(t_1, t_2) = \text{ATT}(t_2) = \text{ATT}(t^*) \cdot \rho^{\,t_2 - t^*}$$

**Bias of the standard estimator**:

$$\text{Bias} = \text{DiD}(t_1, t_2) - \text{ATT}(t^*) = \text{ATT}(t^*)\bigl(\rho^{t_2 - t^*} - 1\bigr)$$

| $\rho$ | Dynamics | Sign of bias |
|--------|----------|--------------|
| $< 1$ | Decaying | Negative — standard DiD **under-estimates** ATT$(t^*)$ |
| $= 1$ | Constant | Zero — standard DiD is **unbiased** |
| $> 1$ | Growing  | Positive — standard DiD **over-estimates** ATT$(t^*)$ |

**AR(1)-adjusted estimator** (point-identified given $\rho$):

$$\widehat{\text{ATT}}(t^*) = \frac{\text{DiD}(t_1, t_2)}{\rho^{\,t_2 - t^*}}$$

The sensitivity plot displays $\widehat{\text{ATT}}(t^*)$ over a grid of $\rho \in [0.5, 1.5]$. A flat curve near $\rho = 1$ indicates robustness to the assumed dynamics.

---

## Monotone Lower Bound

Without parametric assumptions, we can still partially identify $\text{ATT}(t^*)$.

**Proposition.** Suppose treatment effects are *non-increasing* after the shock:

$$\text{ATT}(t) \leq \text{ATT}(t^*) \quad \text{for all } t \geq t^*$$

Then:

$$\text{DiD}(t_1, t_2) = \text{ATT}(t_2) \leq \text{ATT}(t^*)$$

The standard DiD is a valid **lower bound** on ATT$(t^*)$, and this bound is **sharp** (attained at $\rho = 1$).

The monotonicity assumption is natural for many historical policy shocks, where the immediate implementation effect is largest and subsequent adjustment reduces the treatment contrast over time.

---

## Summary of Strategies

| Strategy | Assumption | Object identified |
|----------|------------|-------------------|
| `standard` | Parallel trends | ATT$(t_2)$ |
| `ar1_adjusted` | AR(1) dynamics with known $\rho$ | ATT$(t^*)$ |
| `monotone_lb` | Parallel trends + non-increasing effects | Lower bound on ATT$(t^*)$ |
