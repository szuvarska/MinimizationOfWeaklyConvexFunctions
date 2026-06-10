"""
Experiment B: Regularized / sparse phase retrieval (TODO section 8.B).

Exercises the r != 0 case of the paper's framework phi = f + r via the proximal
stochastic subgradient method (model 1.4 + prox):

    phi(x) = E |<a, x>^2 - b|  +  mu * ||x||_1,
    x+ = prox_{(mu/beta)||.||_1}( x - (1/beta) v ).

The target xbar is k-sparse (k << d). We compare the L1-regularized method
against the unregularized subgradient method (mu = 0) and show that the L1 prior
recovers the signal from fewer measurements (smaller m/d).

A spectral initialization is used (standard for phase retrieval) so that the
subgradient method reaches the basin of attraction; both methods share the same
init for a fair comparison. Recovery is measured up to a global sign flip.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ModelBasedSolver import ModelBasedSolver
from src.problems.sparse_phase_retrieval import SparsePhaseRetrievalSubgradient
from src.utils import sign_invariant_dist, spectral_init


def generate_sparse_data(d, k, m, seed=0):
    """k-sparse target on the unit sphere; b_i = <a_i, xbar>^2."""
    torch.manual_seed(seed)
    true_x = torch.zeros(d)
    support = torch.randperm(d)[:k]
    true_x[support] = torch.randn(k)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return data, true_x


def _worker(mu, data, d, x_init, beta, n_epochs, m, true_x):
    """Run one trajectory; return (final sign-invariant error, obj curve)."""
    prob = SparsePhaseRetrievalSubgradient(rho=2.0, mu=mu)
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
    err = sign_invariant_dist(x_final, true_x)
    return err, solver.history["obj_values"]


def run_recovery_vs_m(d, k, ratios, beta_inv, mu, n_epochs, n_rounds, data_seed=42):
    print(f"\n[B] recovery vs m/d: d={d}, k={k}, beta^-1={beta_inv}, mu={mu}")
    beta = 1.0 / beta_inv
    methods = {"L1 (sparse)": mu, "No reg": 0.0}
    results = {name: np.zeros(len(ratios)) for name in methods}

    n_workers = max(1, os.cpu_count() - 1)
    tasks = []
    for ri, ratio in enumerate(ratios):
        m = int(round(ratio * d))
        for r in range(n_rounds):
            data, true_x = generate_sparse_data(d, k, m, seed=data_seed + 1000 * r)
            x0 = spectral_init(data, d)
            for name, mu_val in methods.items():
                tasks.append((ri, name, mu_val, data, x0, m, true_x))

    raw = {(ri, name): [] for ri in range(len(ratios)) for name in methods}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker, mu_val, data, d, x0, beta, n_epochs, m, true_x): (
                ri,
                name,
            )
            for ri, name, mu_val, data, x0, m, true_x in tasks
        }
        for fut in as_completed(futures):
            ri, name = futures[fut]
            err, _ = fut.result()
            raw[(ri, name)].append(err)

    for ri in range(len(ratios)):
        for name in methods:
            results[name][ri] = np.mean(raw[(ri, name)])
    return np.array(ratios, dtype=float), results


def run_convergence(d, k, ratio, beta_inv, mu, n_epochs, n_rounds, data_seed=42):
    """Objective-gap curves with vs without the L1 prior at one m/d."""
    print(f"\n[B] convergence at m/d={ratio}")
    beta = 1.0 / beta_inv
    m = int(round(ratio * d))
    methods = {"L1 (sparse)": mu, "No reg": 0.0}

    n_workers = max(1, os.cpu_count() - 1)
    tasks = []
    for r in range(n_rounds):
        data, true_x = generate_sparse_data(d, k, m, seed=data_seed + 1000 * r)
        x0 = spectral_init(data, d)
        for name, mu_val in methods.items():
            tasks.append((name, mu_val, data, x0, m, true_x))

    raw = {name: [] for name in methods}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker, mu_val, data, d, x0, beta, n_epochs, m, true_x): name
            for name, mu_val, data, x0, m, true_x in tasks
        }
        for fut in as_completed(futures):
            _, obj = fut.result()
            raw[futures[fut]].append(obj)

    curves = {}
    for name in methods:
        min_len = min(len(c) for c in raw[name])
        curves[name] = np.mean([c[:min_len] for c in raw[name]], axis=0)
    return m, curves


def plot_b(ratios, recovery, conv_m, conv_curves, d, k, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, errs in recovery.items():
        ax1.plot(ratios, errs, marker="o", linewidth=1.5, label=name)
    ax1.set_xlabel("Oversampling ratio $m/d$")
    ax1.set_ylabel(r"recovery error $\min(\|x-\bar x\|, \|x+\bar x\|)$")
    ax1.set_yscale("log")
    ax1.set_title(f"Sparse recovery — d={d}, k={k}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name, curve in conv_curves.items():
        ax2.plot(
            np.arange(len(curve)), np.maximum(curve, 1e-12), linewidth=1.5, label=name
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"function gap $\varphi_{\mathrm{fit}}(x_t)$")
    ax2.set_yscale("log")
    ax2.set_title(f"Convergence at m/d = {conv_m / d:.2g}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"sparse_phase_retrieval_d{d}_k{k}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d, k = 200, 10
    ratios = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    beta_inv, mu = 0.02, 0.1
    n_epochs, n_rounds = 200, 8

    ratios_arr, recovery = run_recovery_vs_m(
        d, k, ratios, beta_inv, mu, n_epochs, n_rounds
    )
    conv_m, conv_curves = run_convergence(
        d, k, ratio=1.0, beta_inv=beta_inv, mu=mu, n_epochs=n_epochs, n_rounds=n_rounds
    )
    plot_b(ratios_arr, recovery, conv_m, conv_curves, d, k, save_dir)
    print("\nSparse phase retrieval experiment complete.")


if __name__ == "__main__":
    main()
