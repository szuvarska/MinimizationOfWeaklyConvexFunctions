"""
Experiment E: Statistical phase transition (TODO section 8.E).

Empirical recovery probability vs the oversampling ratio m/d for phase retrieval.
For each ratio we run many independent trials (fresh data + random init on the
unit sphere) and count a trial as a success when the sign-invariant recovery
error falls below a tolerance after a fixed budget. The three model-based methods
are overlaid; the prox-point / prox-linear methods exhibit a sharp threshold at a
smaller m/d than the subgradient method.
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
from src.utils import sign_invariant_dist

METHOD_CLASSES = {
    "Subgradient": SubgradientPhaseRetrieval,
    "Prox-Linear": ProxLinearPhaseRetrieval,
    "Proximal Point": ProximalPointPhaseRetrieval,
}


def generate_trial(d, m, seed):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)
    return data, true_x, x_init


def _worker(method_name, d, m, beta, n_epochs, tol, seed):
    data, true_x, x_init = generate_trial(d, m, seed)
    prob = METHOD_CLASSES[method_name](rho=2.0)
    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=n_epochs * m,
        batch_size=1,
        beta=beta,
        log_every=m,
        verbose=False,
    )
    x_final = solver.run()
    return 1.0 if sign_invariant_dist(x_final, true_x) < tol else 0.0


def run_phase_transition(d, ratios, beta_inv, n_epochs, n_trials, tol=1e-3):
    print(f"\n[E] phase transition: d={d}, beta^-1={beta_inv}, {n_trials} trials/ratio")
    beta = 1.0 / beta_inv
    results = {name: np.zeros(len(ratios)) for name in METHOD_CLASSES}

    tasks = []
    for ri, ratio in enumerate(ratios):
        m = int(round(ratio * d))
        for name in METHOD_CLASSES:
            for t in range(n_trials):
                tasks.append((ri, name, m, 100 * ri + t))

    n_workers = max(1, os.cpu_count() - 1)
    raw = {(ri, name): [] for ri in range(len(ratios)) for name in METHOD_CLASSES}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker, name, d, m, beta, n_epochs, tol, seed): (ri, name)
            for ri, name, m, seed in tasks
        }
        for fut in as_completed(futures):
            ri, name = futures[fut]
            raw[(ri, name)].append(fut.result())

    for ri in range(len(ratios)):
        for name in METHOD_CLASSES:
            results[name][ri] = np.mean(raw[(ri, name)])
    return np.array(ratios, dtype=float), results


def plot_e(ratios, results, d, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, probs in results.items():
        ax.plot(ratios, probs, marker="o", linewidth=1.6, label=name)
    ax.set_xlabel("Oversampling ratio $m/d$")
    ax.set_ylabel("Empirical recovery probability")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Phase transition — d = {d}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"phase_transition_d{d}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d = 50
    ratios = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    beta_inv = 0.001
    n_epochs, n_trials = 200, 50

    ratios_arr, results = run_phase_transition(d, ratios, beta_inv, n_epochs, n_trials)
    plot_e(ratios_arr, results, d, save_dir)
    print("\nPhase transition experiment complete.")


if __name__ == "__main__":
    main()
