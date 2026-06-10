# Project TODO

## Goal

Reproduce the experiments from Davis & Drusvyatskiy (2019) — comparing three stochastic model-based methods on phase retrieval and blind deconvolution, then additionally compare with standard optimizers (SGD, Adam, AdaGrad).

---

## 1. Fix Existing Bugs

- [x] Remove duplicate `population_data[:, 1] += 4.0` line in notebooks 05 and 06
- [x] Fix `solve_exact` to handle full mini-batch (currently only uses `batch[0]`)

## 2. Implement the Three Methods from the Paper

All three solve: `x_{t+1} = argmin { model(x) + r(x) + (β_t/2)||x - x_t||² }`

- [x] **Stochastic subgradient** (model 1.4): `f_x(y,ξ) = f(x) + <G(x,ξ), y-x>`
- [x] **Stochastic prox-linear** (model 1.5): `f_x(y,ξ) = h(c(x,ξ) + ∇c(x,ξ)(y-x), ξ)`
  - Closed-form for phase retrieval: eq. (5.2)
  - Closed-form for blind deconvolution: same formula with ζ = λ(<v,y>u, <u,x>v)
- [x] **Stochastic proximal point** (model 1.6): `f_x(y,ξ) = h(c(y,ξ))`
  - Closed-form for phase retrieval: eq. (5.3)-(5.4), enumerate 4 candidate solutions
  - Closed-form for blind deconvolution: eq. (5.5)-(5.6), solve quartic

## 3. Implement Convergence Tracking

- [x] Track **Moreau envelope gradient norm** `||∇φ_λ(x)||` — the paper's main convergence measure
- [x] Track function value gap `φ(x_t) - φ(x*)`
- [x] Track distance to true solution `||x_t - x̄||`

## 4. Experiment 1: Phase Retrieval (Section 5.1)

Setup:

- Generate Gaussian measurements `a_i ~ N(0, I_d)`, i=1,...,m
- Generate target `x̄` and initial `x_0` uniformly on unit sphere
- Set `b_i = <a_i, x̄>²`
- Objective: `min (1/m) Σ |<a_i, x>² - b_i|`

Configurations (same as paper):

- [x] (d, m) = (10, 30)
- [x] (d, m) = (50, 150)
- [x] (d, m) = (100, 300)

For each config:

- [x] Run all 3 methods with 100 equally spaced step-sizes β_t⁻¹ ∈ [10⁻⁴, 1]
- [x] Plot function gap after 100 passes through data (averaged over 15 rounds)
- [x] Plot epochs to reach 10⁻⁴ functional suboptimality (prox-linear & prox-point)

Script: `experiments/phase_retrieval.py` (smoke-tested, ready to run full)

## 5. Experiment 2: Blind Deconvolution (Section 5.2)

Setup:

- Generate `u_i ~ N(0, I_d1)`, `v_i ~ N(0, I_d2)`, i=1,...,m
- Generate target `x̄` uniformly on unit sphere
- Set `b_i = <u_i, x̄><v_i, x̄>`
- Objective: `min (1/m) Σ |<u_i, x><v_i, y> - b_i|`

Configurations (same as paper):

- [x] (d1, d2, m) = (10, 10, 30)
- [x] (d1, d2, m) = (50, 50, 200)
- [x] (d1, d2, m) = (100, 100, 400)

For each config:

- [x] Same plots as phase retrieval experiment

Script: `experiments/blind_deconvolution.py` (smoke-tested, ready to run full)

## 6. Additional Comparison with Standard Optimizers

- [x] Implement SGD baseline (torch.optim.SGD on same loss with autograd)
- [x] Implement Adam baseline (torch.optim.Adam)
- [x] Implement AdaGrad baseline (torch.optim.Adagrad)
- [x] Compare convergence curves: model-based methods vs standard optimizers
- [x] Show step-size sensitivity advantage of prox-linear/prox-point

Script: `experiments/optimizer_comparison.py` (smoke-tested, ready to run full)

## 7. Deliverables

- [x] Clean experiment scripts in `experiments/`
- [ ] Run full experiments and generate final plots (Figure 3, Figure 4, optimizer comparison)
- [ ] Presentation/report in `deliverables/`

---

## 8. New Experiments (A–E) + Stop Criteria (F)

Extensions beyond the paper's Figures 3–4. Each item lists the paper hook, data,
what to track, the plot, new code, and expected result. Effort: S / M / L.

### Shared infrastructure (do first — reused by B, D, E, F)

- [ ] `src/utils.py`: `sign_invariant_dist(x, x_ref) = min(||x - x_ref||, ||x + x_ref||)`
      — phase retrieval recovers `x̄` only up to global sign. Used by B, D, E.
- [ ] `src/regularizers.py`: `l1_prox(x, thr)` soft-threshold
      `sign(x) * relu(|x| - thr)` (prox of `thr * ||·||_1`). Used by B.
- [ ] `ModelBasedSolver`: optional early stop — params `tol`, `patience`,
      `stop_measure ∈ {"moreau", "prox_step", "none"}`; record `self.stop_iter`.
      Used by F. (Keep default `"none"` so existing experiments are unaffected.)

### A. Verify the O(k^-1/4) convergence rate — Effort: S

Paper: headline result — stationarity measure `min_t E||∇φ_λ(x_t)||` decays at `O(T^-1/4)`
(abstract; §4 theorems). Equivalently `min_t E||∇φ_λ||² = O(T^-1/2)`.

- [ ] Reuse `compute_moreau_grad_norm` (λ = 1/(2ρ)) from `WeaklyConvexProblem`.
- [ ] **A1 (single long run):** small constant `β`; plot running-min
      `min_{s≤t} ||∇φ_λ(x_s)||` vs `t` on log-log with a reference slope `-1/4`.
      Expect decay to a noise floor.
- [ ] **A2 (horizon sweep — the rigorous test):** for `T ∈ {3e2,1e3,3e3,1e4,3e4,1e5}`
      set `β = c·√T` (paper's `α ∝ 1/√T`); record `min_t ||∇φ_λ(x_t)||`; plot vs `T`
      log-log; fit slope, expect ≈ `-1/4`.
- [ ] Config: phase retrieval (d,m) = (50,150), ≥10 rounds, all 3 methods.
- [ ] Output: `deliverables/figures/convergence_rate_d50_m150.png`.
- [ ] New file: `experiments/convergence_rate.py`. No solver change.
- [ ] Expected: A2 slope close to `-1/4`; prox-linear/point lower constant than subgradient.

### B. Regularized / sparse phase retrieval (r ≠ 0) — Effort: M

Paper: the full framework is `φ = f + r`; proximal stochastic subgradient applies
`prox_{αr}` after the step (§1, model 1.4). No current experiment uses `r ≠ 0`.

- [ ] Data: k-sparse `x̄` (k ≪ d), `b_i = <a_i, x̄>²`, compressed regime `m ~ O(k log d)`.
- [ ] New problem `SparsePhaseRetrievalSubgradient` in
      `src/problems/sparse_phase_retrieval.py`: subgradient step then soft-threshold,
      `x_{t+1} = prox_{αμ||·||_1}(x_t − α v)` using `l1_prox`.
- [ ] (optional) prox-linear / prox-point with `r` via the existing LBFGS fallback
      (`solve_exact` returns `None`) — note the added per-step cost.
- [ ] Track: sign-invariant recovery error (`sign_invariant_dist`) + iterate sparsity
      (`||x||_0` at tol 1e-3).
- [ ] Plots: (i) recovery error vs `m/d`, sparse-ℓ1 vs dense-no-reg; (ii) convergence
      curves with vs without the regularizer.
- [ ] Config: d=200, k=10, sweep m, ≥10 rounds.
- [ ] Output: `deliverables/figures/sparse_phase_retrieval.png`.
- [ ] New file: `experiments/sparse_phase_retrieval.py`.
- [ ] Expected: ℓ1 recovers at smaller `m/d` than the unregularized dense baseline.

### C. Convex-composite / CVaR — improved O(ε^-2) rate — Effort: M

Paper: when models are convex (τ = 0) the function-value rate improves to `O(ε^-2)`;
`O(1/(με))` if μ-strongly convex (§1 p.5; Example 2.6; §4.2). `max_parabola.py` already
has the model but no experiment.

- [ ] Wrap `max_parabola_phi` / `max_parabola_model_gen` ([max_parabola.py](src/problems/max_parabola.py))
      in `MaxParabolaProblem(WeaklyConvexProblem)` — max of parabolas is convex ⇒ τ = 0.
- [ ] Compute population optimum `φ*` once (fine grid / LBFGS) for the gap.
- [ ] Run prox-linear (model 1.5) + subgradient via the solver's numerical path.
- [ ] **Convex variant:** plot function gap `φ(x_t) − φ*` vs `k` log-log, reference slope `-1/2`.
- [ ] **Strongly convex variant:** add `(μ/2)||x||²`; expect slope `-1` (rate `1/(μk)`).
- [ ] Config: d ∈ {1, 10}, ≥15 rounds.
- [ ] Output: `deliverables/figures/cvar_convex_rate.png`.
- [ ] New file: `experiments/cvar_convex.py`.
- [ ] Expected: convex slope ≈ `-1/2`, strongly convex ≈ `-1`.

### D. Robustness to outliers (ℓ1 vs ℓ2) — Effort: M

Paper: Example 2.1 — the ℓ1 penalty gives recovery/stability under "gross outliers."

- [ ] Data: phase retrieval; corrupt fraction `p` of `b_i` with large random values.
      Sweep `p ∈ {0, .05, .1, .2, .3}`.
- [ ] Robust: ℓ1 `|<a,x>² − b|` via prox-point / prox-linear (existing classes).
- [ ] Baseline: smooth ℓ2 `(<a,x>² − b)²` (Wirtinger-flow-like) via `torch.optim`
      (SGD / Adam) — add `f_phase_retrieval_l2` in the experiment file.
- [ ] Metric: sign-invariant recovery error vs `p`, ≥15 rounds, best stepsize per method.
- [ ] Plot: recovery error vs outlier fraction `p` — robust-ℓ1 vs ℓ2.
- [ ] Config: (d,m) = (50,300) (oversampled so clean recovery is easy at p=0).
- [ ] Output: `deliverables/figures/outlier_robustness_d50_m300.png`.
- [ ] New file: `experiments/outlier_robustness.py`.
- [ ] Expected: ℓ1 stays low as `p` grows; ℓ2 degrades sharply.

### E. Statistical phase transition (recovery vs m/d) — Effort: M

Paper: sample-complexity discussion (`O(ε^-4)`); classic phase-retrieval threshold.

- [ ] Fix d=50; sweep `m` so `m/d ∈ {1, 1.5, 2, ..., 6}`.
- [ ] Per ratio: N=50 trials (fresh data + init); success = sign-invariant error < 1e-3
      after a fixed budget.
- [ ] Method: prox-point (best); optionally overlay prox-linear + subgradient.
- [ ] Plot: empirical recovery probability vs `m/d` (expect sharp S-curve, threshold ~2–3).
- [ ] Output: `deliverables/figures/phase_transition_d50.png`.
- [ ] New file: `experiments/phase_transition.py`.
- [ ] Expected: sharp transition; prox-point transitions at the smallest `m/d`.

### F. Stop criteria via the Moreau envelope — Effort: M

Why Moreau: `||∇φ_λ(x)||` is the computable stationarity measure; small value ⇒ `x` is
near `x̂ = prox_{λφ}(x)` with `dist(0; ∂φ(x̂)) ≤ ||∇φ_λ(x)||` and `φ(x̂) ≤ φ(x)` — certifies
near-stationarity **without knowing φ\***. Since `φ* = 0` here, the function-gap rule is a
ground-truth reference for validation.

- [ ] Solver early stop (see Shared infra): check `||∇φ_λ||` every `moreau_every`; stop
      after `patience` consecutive checks below `tol`; record `stop_iter`. `prox_step`
      mode uses the ~free proxy `β||x_{t+1} − x_t||`.
- [ ] Experiment comparing 4 rules on one phase-retrieval run:
      1. Moreau gradient norm ≤ ε  (principled, no φ\* needed)
      2. Function gap ≤ ε          (oracle reference, only because φ\* = 0)
      3. Iterate stagnation `||x_{t+1} − x_t|| ≤ ε`  (cheap)
      4. Fixed budget
- [ ] Show: Moreau-stop fires at a near-optimal point and tracks the oracle gap-stop;
      report a table of (stop epoch, final error, # extra Moreau solves); mark each rule
      on the convergence curve.
- [ ] Config: (d,m) = (50,150), good stepsize `β^-1 = 0.1`, ≥5 rounds.
- [ ] Output: `deliverables/figures/stop_criteria_d50_m150.png`.
- [ ] New file: `experiments/stop_criteria.py`.
- [ ] Expected: Moreau-stop ≈ oracle gap-stop; stagnation stops too early; fixed budget wastes epochs.
