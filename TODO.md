# Project TODO

## Goal

Reproduce the experiments from Davis & Drusvyatskiy (2019) — comparing three stochastic model-based methods on phase retrieval and blind deconvolution, then additionally compare with standard optimizers (SGD, Adam, AdaGrad).

---

## 1. Fix Existing Bugs

- [x] Remove duplicate `population_data[:, 1] += 4.0` line in notebooks 05 and 06
- [x] Fix `solve_exact` to handle full mini-batch (currently only uses `batch[0]`)

## 2. Implement the Three Methods from the Paper

All three solve: `x_{t+1} = argmin { model(x) + r(x) + (β_t/2)||x - x_t||² }`

- [ ] **Stochastic subgradient** (model 1.4): `f_x(y,ξ) = f(x) + <G(x,ξ), y-x>`
- [ ] **Stochastic prox-linear** (model 1.5): `f_x(y,ξ) = h(c(x,ξ) + ∇c(x,ξ)(y-x), ξ)`
  - Closed-form for phase retrieval: eq. (5.2)
  - Closed-form for blind deconvolution: same formula with ζ = λ(<v,y>u, <u,x>v)
- [ ] **Stochastic proximal point** (model 1.6): `f_x(y,ξ) = h(c(y,ξ))`
  - Closed-form for phase retrieval: eq. (5.3)-(5.4), enumerate 4 candidate solutions
  - Closed-form for blind deconvolution: eq. (5.5)-(5.6), solve quartic

## 3. Implement Convergence Tracking

- [x] Track **Moreau envelope gradient norm** `||∇φ_λ(x)||` — the paper's main convergence measure
- [x] Track function value gap `φ(x_t) - φ(x*)`
- [ ] Track distance to true solution `||x_t - x̄||`

## 4. Experiment 1: Phase Retrieval (Section 5.1)

Setup:

- Generate Gaussian measurements `a_i ~ N(0, I_d)`, i=1,...,m
- Generate target `x̄` and initial `x_0` uniformly on unit sphere
- Set `b_i = <a_i, x̄>²`
- Objective: `min (1/m) Σ |<a_i, x>² - b_i|`

Configurations (same as paper):

- [ ] (d, m) = (10, 30)
- [ ] (d, m) = (50, 150)
- [ ] (d, m) = (100, 300)

For each config:

- [ ] Run all 3 methods with 100 equally spaced step-sizes β_t⁻¹ ∈ [10⁻⁴, 1]
- [ ] Plot function gap after 100 passes through data (averaged over 15 rounds)
- [ ] Plot epochs to reach 10⁻⁴ functional suboptimality (prox-linear & prox-point)

## 5. Experiment 2: Blind Deconvolution (Section 5.2)

Setup:

- Generate `u_i ~ N(0, I_d1)`, `v_i ~ N(0, I_d2)`, i=1,...,m
- Generate target `x̄` uniformly on unit sphere
- Set `b_i = <u_i, x̄><v_i, x̄>`
- Objective: `min (1/m) Σ |<u_i, x><v_i, y> - b_i|`

Configurations (same as paper):

- [ ] (d1, d2, m) = (10, 10, 30)
- [ ] (d1, d2, m) = (50, 50, 200)
- [ ] (d1, d2, m) = (100, 100, 400)

For each config:

- [ ] Same plots as phase retrieval experiment

## 6. Additional Comparison with Standard Optimizers

- [ ] Implement SGD baseline (torch.optim.SGD on same loss with autograd)
- [ ] Implement Adam baseline (torch.optim.Adam)
- [ ] Implement AdaGrad baseline (torch.optim.Adagrad)
- [ ] Compare convergence curves: model-based methods vs standard optimizers
- [ ] Show step-size sensitivity advantage of prox-linear/prox-point

## 7. Deliverables

- [ ] Clean experiment scripts in `experiments/`
- [ ] Final plots reproducing paper's Figure 3 and Figure 4
- [ ] Presentation/report in `deliverables/`
