"""
Experiment A: Verify the O(k^-1/4) convergence rate (TODO section 8.A).

The paper's headline guarantee is that the stationarity measure
    min_{t<=k} ||grad phi_lambda(x_t)||
tends to zero at the rate O(k^-1/4)  (equivalently min_t ||.||^2 = O(k^-1/2)).

A1 (single long run): one trajectory at a small constant beta; plot the running
    minimum of ||grad phi_lambda|| vs iteration on log-log with a -1/4 slope line.
    Expect a -1/4 decay until a stochastic noise floor.

A2 (horizon sweep, the rigorous test): for several horizons T, set
    beta = beta_scale * sqrt(T)  (the paper's alpha ~ 1/sqrt(T)); record
    min_t ||grad phi_lambda(x_t)|| and plot vs T on log-log. A least-squares fit
    with slope ~ -1/4 confirms the theorem.

Moreau gradient norm uses lambda = 1/(2 rho) via WeaklyConvexProblem.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ModelBasedSolver import ModelBasedSolver
from src.problems.phase_retrieval import (
    SubgradientPhaseRetrieval,
    ProxLinearPhaseRetrieval,
    ProximalPointPhaseRetrieval,
)


def generate_phase_retrieval_data(d, m, seed=0):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    population_data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return population_data, true_x


METHOD_CLASSES = {
    "Subgradient": SubgradientPhaseRetrieval,
    "Prox-Linear": ProxLinearPhaseRetrieval,
    "Proximal Point": ProximalPointPhaseRetrieval,
}


def run_trajectory(method_name, data, d, beta, T, moreau_every, seed):
    """Run one trajectory; return list of (iteration, Moreau grad norm)."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)

    prob = METHOD_CLASSES[method_name](rho=2.0)
    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=T,
        batch_size=1,
        beta=beta,
        log_every=moreau_every,
        moreau_every=moreau_every,
        verbose=False,
    )
    solver.run()
    return solver.history["moreau_grad_norms"]


def _worker_a1(method_name, data, d, beta, T, moreau_every, seed):
    return run_trajectory(method_name, data, d, beta, T, moreau_every, seed)


def _worker_a2(method_name, data, d, beta, T, moreau_every, seed):
    pairs = run_trajectory(method_name, data, d, beta, T, moreau_every, seed)
    vals = [v for _, v in pairs]
    return min(vals) if vals else float("inf")


def run_a1(d, m, beta_inv, T, moreau_samples, n_rounds, data_seed=42):
    print(f"\n[A1] single-run running-min: d={d}, m={m}, beta^-1={beta_inv}, T={T}")
    data, _ = generate_phase_retrieval_data(d, m, seed=data_seed)
    beta = 1.0 / beta_inv
    moreau_every = max(1, T // moreau_samples)

    tasks = [(name, r) for name in METHOD_CLASSES for r in range(n_rounds)]
    n_workers = max(1, os.cpu_count() - 1)
    raw = {name: [] for name in METHOD_CLASSES}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker_a1, name, data, d, beta, T, moreau_every, r): name
            for name, r in tasks
        }
        for fut in as_completed(futures):
            raw[futures[fut]].append(fut.result())

    results = {}
    for name in METHOD_CLASSES:
        curves = raw[name]
        iters = [it for it, _ in curves[0]]
        min_len = min(len(c) for c in curves)
        iters = iters[:min_len]
        avg = np.mean([[v for _, v in c][:min_len] for c in curves], axis=0)
        running_min = np.minimum.accumulate(avg)
        results[name] = (np.array(iters), running_min)
    return results


def run_a2(d, m, horizons, beta_scale, moreau_samples, n_rounds, data_seed=42):
    print(f"\n[A2] horizon sweep: d={d}, m={m}, beta={beta_scale}*sqrt(T)")
    data, _ = generate_phase_retrieval_data(d, m, seed=data_seed)

    tasks = []
    for name in METHOD_CLASSES:
        for T in horizons:
            beta = beta_scale * np.sqrt(T)
            moreau_every = max(1, T // moreau_samples)
            for r in range(n_rounds):
                tasks.append((name, T, beta, moreau_every, r))

    n_workers = max(1, os.cpu_count() - 1)
    raw = {(name, T): [] for name in METHOD_CLASSES for T in horizons}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker_a2, name, data, d, beta, T, moreau_every, r): (name, T)
            for name, T, beta, moreau_every, r in tasks
        }
        for fut in as_completed(futures):
            raw[futures[fut]].append(fut.result())

    results = {}
    for name in METHOD_CLASSES:
        mins = np.array([np.mean(raw[(name, T)]) for T in horizons])
        results[name] = mins
    return np.array(horizons, dtype=float), results


def plot_rate(a1_results, horizons, a2_results, d, m, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # A1: running-min vs iteration with -1/4 reference slope
    for name, (iters, rmin) in a1_results.items():
        mask = iters > 0
        ax1.plot(
            iters[mask], rmin[mask], label=name, linewidth=1.5, marker="o", markersize=3
        )
    ref_x = np.array([it for it in a1_results["Subgradient"][0] if it > 0], dtype=float)
    if len(ref_x):
        anchor = a1_results["Subgradient"][1][a1_results["Subgradient"][0] > 0][0]
        ref_y = anchor * (ref_x / ref_x[0]) ** (-0.25)
        ax1.plot(ref_x, ref_y, "k--", alpha=0.6, label=r"slope $-1/4$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration $k$")
    ax1.set_ylabel(r"$\min_{t\leq k}\,\|\nabla\varphi_\lambda(x_t)\|$")
    ax1.set_title(f"A1: running-min stationarity — (d, m) = ({d}, {m})")
    ax1.legend()
    ax1.grid(True, alpha=0.3, which="both")

    # A2: min stationarity vs horizon T with fitted slopes
    for name, mins in a2_results.items():
        slope = np.polyfit(np.log(horizons), np.log(mins), 1)[0]
        ax2.plot(
            horizons,
            mins,
            label=f"{name} (slope {slope:.2f})",
            linewidth=1.5,
            marker="s",
            markersize=5,
        )
    ref_y = a2_results["Subgradient"][0] * (horizons / horizons[0]) ** (-0.25)
    ax2.plot(horizons, ref_y, "k--", alpha=0.6, label=r"slope $-1/4$")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Horizon $T$  ($\\beta \\propto \\sqrt{T}$)")
    ax2.set_ylabel(r"$\min_t\,\|\nabla\varphi_\lambda(x_t)\|$")
    ax2.set_title(f"A2: horizon sweep — (d, m) = ({d}, {m})")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"convergence_rate_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d, m = 50, 150

    a1 = run_a1(d, m, beta_inv=0.05, T=30000, moreau_samples=40, n_rounds=10)
    horizons = [300, 1000, 3000, 10000, 30000]
    hz, a2 = run_a2(d, m, horizons, beta_scale=1.0, moreau_samples=30, n_rounds=10)
    plot_rate(a1, hz, a2, d, m, save_dir)
    print("\nConvergence-rate experiment complete.")


if __name__ == "__main__":
    main()
