"""
Experiment C: Convex vs strongly-convex function-value rate (TODO section 8.C).

Paper (Davis & Drusvyatskiy 2019, p.5): when the models are true under-estimators
(tau = 0) and f_x + r is convex, Algorithm 1.2 attains function-value complexity
O(eps^-2); if mu-strongly convex, it improves to O(1/(mu*eps)).

We use the convex composite instance robust linear regression
    phi(x) = E |<a, x> - b|  +  (mu/2) ||x||^2,   b = <a, xbar>,
which is convex for mu = 0 and mu-strongly convex for mu > 0 (a CVaR loss is
another such convex-composite instance). The stochastic subgradient method is
run with the paper's control sequences:
    * convex          : beta_t = c * sqrt(t+1)   (alpha_t ~ 1/sqrt(t))  -> slope -1/2
    * strongly convex : beta_t = mu * (t+1)      (alpha_t ~ 1/(mu t))   -> slope -1

We plot the running-min function gap phi(x_t) - phi* vs iteration on log-log
with -1/2 and -1 reference slopes.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ModelBasedSolver import ModelBasedSolver
from src.problems.robust_regression import (
    RobustRegressionSubgradient,
    f_robust_regression,
)


def generate_regression_data(d, m, seed=0):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = a_matrix @ true_x
    data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return data, true_x


def compute_phi_star(data, mu, d, n_restarts=3, max_iter=500):
    """Minimize the full-population objective once to get phi* (gap reference)."""
    best = float("inf")
    for r in range(n_restarts):
        torch.manual_seed(1000 + r)
        x = torch.randn(d, requires_grad=True)
        opt = torch.optim.LBFGS(
            [x], lr=1.0, max_iter=max_iter, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            loss = torch.mean(f_robust_regression(x, data)) + 0.5 * mu * torch.sum(x**2)
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            val = (
                torch.mean(f_robust_regression(x, data)) + 0.5 * mu * torch.sum(x**2)
            ).item()
            best = min(best, val)
    return best


def _worker(setting, data, d, mu, c, T, m, seed):
    """Run one subgradient trajectory; return per-epoch objective values."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)

    prob = RobustRegressionSubgradient(rho=0.0, mu=mu)

    if setting == "convex":

        def beta_fn(t, c=c):
            return c * np.sqrt(t + 1)

    else:  # strongly convex

        def beta_fn(t, mu=mu):
            return mu * (t + 1)

    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=T,
        batch_size=1,
        beta=beta_fn,
        log_every=m,
        verbose=False,
    )
    solver.run()
    return solver.history["obj_values"]


def run_setting(setting, d, m, mu, c, T, n_rounds, data_seed=42):
    print(f"\n[{setting}] d={d}, m={m}, mu={mu}, T={T}")
    data, _ = generate_regression_data(d, m, seed=data_seed)
    phi_star = compute_phi_star(data, mu, d) if mu > 0 else 0.0
    print(f"  phi* = {phi_star:.6e}")

    n_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_worker, setting, data, d, mu, c, T, m, r)
            for r in range(n_rounds)
        ]
        curves = [f.result() for f in as_completed(futures)]

    min_len = min(len(c_) for c_ in curves)
    avg_obj = np.mean([c_[:min_len] for c_ in curves], axis=0)
    gap = np.maximum(avg_obj - phi_star, 1e-12)
    running_min = np.minimum.accumulate(gap)
    epochs = np.arange(min_len)
    return epochs, running_min


def plot_rates(convex, strongly, d, m, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))

    ep_c, gap_c = convex
    ep_s, gap_s = strongly
    ax.plot(ep_c[1:], gap_c[1:], "C0-", linewidth=1.6, label="Convex ($\\mu=0$)")
    ax.plot(ep_s[1:], gap_s[1:], "C1-", linewidth=1.6, label="Strongly convex")

    # Reference slopes anchored at the first plotted point of each curve
    x_c = ep_c[1:].astype(float)
    ax.plot(
        x_c,
        gap_c[1] * (x_c / x_c[0]) ** (-0.5),
        "C0--",
        alpha=0.6,
        label=r"slope $-1/2$",
    )
    x_s = ep_s[1:].astype(float)
    ax.plot(
        x_s, gap_s[1] * (x_s / x_s[0]) ** (-1.0), "C1--", alpha=0.6, label=r"slope $-1$"
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch $k$")
    ax.set_ylabel(r"running-min gap $\varphi(x_k) - \varphi^*$")
    ax.set_title(f"Convex vs strongly-convex rate — (d, m) = ({d}, {m})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"convex_strongly_convex_rate_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d, m = 10, 200
    T = 200 * m  # 200 epochs
    n_rounds = 15

    convex = run_setting("convex", d, m, mu=0.0, c=2.0, T=T, n_rounds=n_rounds)
    strongly = run_setting("strongly", d, m, mu=0.5, c=2.0, T=T, n_rounds=n_rounds)
    plot_rates(convex, strongly, d, m, save_dir)
    print("\nConvex/strongly-convex rate experiment complete.")


if __name__ == "__main__":
    main()
