# Stochastic Minimization of Weakly Convex Functions — Experimental Report

**Project 5.3** · Based on Davis, D., & Drusvyatskiy, D. (2019). *Stochastic
model-based minimization of weakly convex functions.* SIAM Journal on
Optimization, 29(1), 207–239.

---

## 1. Introduction

Many important problems in signal processing and machine learning — phase
retrieval, blind deconvolution, robust regression, and risk-averse learning —
lead to optimization problems that are **nonsmooth and nonconvex**, yet share a
benign structure called *weak convexity*: a function $\varphi$ is
$\rho$-weakly convex if $\varphi(\cdot) + \tfrac{\rho}{2}\lVert\cdot\rVert^2$ is
convex. Classical stochastic gradient theory does not directly apply to such
functions, so for a long time there was no clean convergence guarantee for the
simple iterative methods practitioners actually use.

Davis and Drusvyatskiy (2019) close this gap. They analyze a family of
**stochastic model-based algorithms** — which includes the stochastic
subgradient, prox-linear, and proximal-point methods as special cases — through
the lens of the **Moreau envelope**. Their key result is that the gradient norm
of the Moreau envelope, a principled measure of *near-stationarity*, decays at
the rate $O(k^{-1/4})$, and improves to $O(k^{-1/2})$ (convex models) or
$O(k^{-1})$ (strongly convex models).

This report documents our reproduction of the paper's core experiments and a set
of additional experiments that probe the theory from angles the original paper
does not measure directly: the empirical convergence *rate*, the effect of
regularization, robustness to outliers, statistical sample-complexity thresholds,
and practical stopping criteria. Section 2 fixes notation and recalls the
algorithm; Section 3 describes the shared experimental setup; Section 4 presents
each experiment with its motivation, setup, details, and expected outcome.

---

## 2. Background

### 2.1 Problem class

We consider composite stochastic problems of the form
$$
\min_{x}\; \varphi(x) = \mathbb{E}_{\xi}\big[f(x,\xi)\big] + r(x),
$$
where each $f(\cdot,\xi)$ is $\rho$-weakly convex and $r$ is a closed convex
(possibly nonsmooth) regularizer. A canonical example is **phase retrieval**,
$f(x,\xi) = \lvert \langle a, x\rangle^2 - b\rvert$ with $b = \langle a,\bar
x\rangle^2$, which is $2\lVert a\rVert^2$-weakly convex.

### 2.2 Stochastic model-based algorithm

At each step the method samples $\xi_t$, builds a simple **model**
$f_{x_t}(\cdot,\xi_t)$ of $f$ around the current iterate, and solves a
proximal subproblem:
$$
x_{t+1} = \operatorname*{arg\,min}_{y}\;\Big\{\, f_{x_t}(y,\xi_t) + r(y) +
\tfrac{\beta_t}{2}\lVert y - x_t\rVert^2 \,\Big\}.
$$
The three methods differ only in the model:

| Method | Model $f_x(y,\xi)$ | Paper eq. |
|---|---|---|
| Subgradient | $f(x,\xi) + \langle v, y-x\rangle,\; v\in\partial f(x,\xi)$ | (1.4) |
| Prox-linear | $h\big(c(x,\xi) + \nabla c(x,\xi)(y-x),\xi\big)$ | (1.5) |
| Proximal point | $f(y,\xi)$ (the function itself) | (1.6) |

The control sequence $\beta_t$ plays the role of an inverse step-size
($\alpha_t = 1/\beta_t$). For phase retrieval and blind deconvolution all three
subproblems have closed-form solutions; otherwise we fall back to an inner LBFGS
solve.

### 2.3 Stationarity measure: the Moreau envelope

For $\lambda < 1/\rho$, the **Moreau envelope**
$\varphi_\lambda(x) = \min_y \{\varphi(y) + \tfrac{1}{2\lambda}\lVert
y-x\rVert^2\}$ is smooth, and its gradient
$$
\nabla\varphi_\lambda(x) = \tfrac{1}{\lambda}\big(x - \operatorname{prox}_{\lambda
\varphi}(x)\big)
$$
satisfies: a small $\lVert\nabla\varphi_\lambda(x)\rVert$ certifies that $x$ is
*close* to a point that is *nearly stationary* for $\varphi$ — **without knowing
the optimum $\varphi^\star$**. This quantity is the paper's central convergence
measure and the object we track throughout. We use $\lambda = 1/(2\rho)$.

---

## 3. Common experimental setup

- **Data.** Gaussian measurement vectors $a_i \sim \mathcal{N}(0, I_d)$; the
  target $\bar x$ lies on the unit sphere. Phase-retrieval rows are
  $[\,a_i \;\; b_i\,]$ with $b_i = \langle a_i,\bar x\rangle^2$.
- **Sampling.** One sample per iteration (batch size 1); one *epoch* is $m$
  iterations.
- **Step-sizes.** Either a constant $\beta$ swept over $\beta^{-1} \in
  [10^{-4}, 1]$, or a schedule $\beta_t$ (e.g. $\beta_t \propto \sqrt{t}$ or
  $\beta_t = \mu t$) for the rate experiments.
- **Averaging & reproducibility.** Each configuration is repeated over several
  random rounds and averaged; runs are parallelized across processes.
- **Sign ambiguity.** Phase retrieval recovers $\bar x$ only up to a global sign,
  so recovery is measured by
  $\operatorname{dist}_\pm(x,\bar x) = \min(\lVert x-\bar x\rVert, \lVert
  x+\bar x\rVert)$.
- **Output.** All figures are written to `deliverables/figures/`. Scripts live in
  `experiments/`; see the README for exact run commands.

---

## 4. Experiments

### Part I — Reproduction of the paper

#### Experiment 1 — Phase retrieval (Figure 3)

- **Why.** Establish that our implementation faithfully reproduces the paper's
  headline comparison of the three methods, and confirm the practical message
  that prox-linear and proximal-point are far less sensitive to the step-size
  than the subgradient method.
- **Setup.** $(d,m) \in \{(10,30),(50,150),(100,300)\}$; $100$ step-sizes
  $\beta^{-1} \in [10^{-4},1]$; $100$ epochs; averaged over $15$ rounds.
- **Details.** Random initialization on the sphere. For each step-size we record
  the function gap after $100$ epochs and the number of epochs to reach $10^{-4}$
  suboptimality.
- **Expected results.** Prox-linear and proximal-point converge across a wide band
  of step-sizes, while the subgradient method only works in a narrow range —
  reproducing the shape of Figure 3.

#### Experiment 2 — Blind deconvolution (Figure 4)

- **Why.** Show the same qualitative behaviour holds for a second, higher-rank
  weakly convex problem, confirming the methods are not phase-retrieval-specific.
- **Setup.** $(d_1,d_2,m) \in \{(10,10,30),(50,50,200),(100,100,400)\}$; same
  step-size grid, epochs, and rounds as Experiment 1.
- **Details.** Rows are $[\,u_i\;\;v_i\;\;b_i\,]$ with the target $[\bar x;\bar y]$
  on the unit sphere; same two metrics as Experiment 1.
- **Expected results.** Curves matching Figure 4: model-based methods robust to
  step-size, subgradient sensitive.

#### Experiment 3 — Comparison with standard optimizers

- **Why.** Place the model-based methods in context against the optimizers
  practitioners default to (SGD, Adam, AdaGrad) and make the step-size-robustness
  advantage explicit.
- **Setup.** Phase retrieval; the three model-based methods versus SGD/Adam/AdaGrad
  on the same loss using autograd, swept over learning rates.
- **Details.** Convergence curves and final accuracy as a function of step-size.
- **Expected results.** Model-based methods remain stable over a broad step-size
  range; SGD/Adam/AdaGrad require careful tuning and diverge or stall outside it.

#### Experiment 4 — Moreau-envelope gradient convergence

- **Why.** Visualize the paper's actual convergence quantity, $\lVert\nabla
  \varphi_\lambda(x_t)\rVert$, decreasing over training — the empirical content of
  Theorems 4.2 / 4.5 / 4.8.
- **Setup.** Phase retrieval; track the Moreau gradient norm at regular epoch
  intervals for each method.
- **Details.** Moreau prox computed by a full-batch inner LBFGS solve;
  $\lambda = 1/(2\rho)$.
- **Expected results.** The norm decreases toward zero for all methods, with
  prox-linear/proximal-point reaching smaller values than the subgradient method.

### Part II — Additional experiments

#### Experiment 5 (A) — Verifying the $O(k^{-1/4})$ rate

- **Why.** Experiment 4 shows the stationarity measure *goes to zero*; here we test
  the stronger, quantitative claim that it does so at the predicted **rate**.
- **Setup.** Phase retrieval $(d,m)=(50,150)$, all three methods, $\ge 10$ rounds.
  Two complementary views:
  - **A1 (single long run):** small constant $\beta$, $T = 3\times10^4$; plot the
    running minimum $\min_{t\le k}\lVert\nabla\varphi_\lambda(x_t)\rVert$.
  - **A2 (horizon sweep):** for each horizon $T \in \{3\mathrm{e}2,\dots,3\mathrm{e}4\}$
    set $\beta \propto \sqrt{T}$ and record the best stationarity along the run.
- **Details.** A2 is the rigorous test because the theory bounds the *best* iterate
  for a horizon-tuned step-size; the A2 curve is plotted log-log.
- **Expected results.** The A2 log-log slope is close to $-1/4$; prox-linear and
  proximal-point sit on a lower constant than the subgradient method.


#### Experiment 9 (E) — Statistical phase transition

- **Why.** Characterize the *sample-complexity threshold* for recovery and compare
  how sharply each method transitions from failure to success.
- **Setup.** Fix $d=50$, sweep $m/d \in \{1,\dots,6\}$; for each ratio run $50$
  independent trials (fresh data and random init); a trial succeeds when the
  sign-invariant recovery error drops below $10^{-3}$.
- **Details.** Empirical recovery probability per ratio for the three methods.
- **Expected results.** A sharp sigmoidal transition; prox-linear/proximal-point
  cross to high recovery probability at a smaller $m/d$ than the subgradient
  method.

#### Experiment 10 (F) — Stopping criteria

- **Why.** A practical question the theory enables: *when can we stop?* Because the
  Moreau gradient norm is computable without $\varphi^\star$, it provides a
  principled stopping certificate. We compare it against common heuristics.
- **Setup.** Phase retrieval $(d,m)=(50,150)$ with frequent logging of (i) the
  Moreau gradient norm, (ii) the function gap (oracle, since $\varphi^\star=0$),
  (iii) the iterate step size $\lVert x_{t+1}-x_t\rVert$, and (iv) a fixed budget.
  We also exercise the solver's built-in early stop (`stop_measure="moreau"`).
- **Details.** Each rule's stop epoch is computed post-hoc on the averaged curves
  and marked on the convergence plot; the built-in stop is reported alongside.
- **Expected results.** The Moreau-based stop fires close to the oracle gap-based
  stop, validating it as a reliable surrogate; the stagnation heuristic is cheap
  but can trigger prematurely or late, and the fixed budget is wasteful when
  convergence is fast.

---

## 5. Reproducing the figures

Each experiment is a standalone script in `experiments/`; run commands and
configurations are listed in the project README under *Running experiments*.
Figures are saved to `deliverables/figures/`.

## 6. Results and discussion

*(To be completed once the full-scale runs are generated. For each experiment,
insert the produced figure, state whether the expected behaviour was observed,
and note any deviations from the theory or from the original paper.)*
