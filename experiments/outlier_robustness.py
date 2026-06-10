"""
Experiment D: Robustness to outliers (TODO section 8.D).

Motivation (Davis & Drusvyatskiy 2019, Example 2.1): the robust L1 phase-retrieval
loss |<a,x>^2 - b| tolerates gross outliers in the measurements b, whereas the
smooth L2 loss (<a,x>^2 - b)^2 does not.

We corrupt a fraction p of the measurements with large values and compare:
  * Robust L1 : stochastic proximal-point method (closed form, model 1.6).
  * Smooth L2 : (<a,x>^2 - b)^2 minimized with Adam (autograd baseline).

Both share a spectral initialization (from the same corrupted data) so the
comparison isolates loss robustness, not initialization. For each method and
outlier level we sweep a few stepsizes and report the best sign-invariant
recovery error, averaged over rounds.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ModelBasedSolver import ModelBasedSolver
from src.problems.phase_retrieval import ProximalPointPhaseRetrieval
from src.utils import sign_invariant_dist, spectral_init


def generate_phase_retrieval_data(d, m, seed=0):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return data, true_x


def corrupt(data, p, seed):
    """Replace a fraction p of measurements b with large outliers."""
    torch.manual_seed(seed)
    data = data.clone()
    b = data[:, -1]
    m = b.shape[0]
    n_bad = int(round(p * m))
    if n_bad > 0:
        idx = torch.randperm(m)[:n_bad]
        scale = 10.0 * b.mean()
        data[idx, -1] = scale * (1.0 + torch.rand(n_bad))
    return data


def _worker_l1(data, d, x_init, beta, n_epochs, m, true_x):
    prob = ProximalPointPhaseRetrieval(rho=2.0)
    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init.clone(),
        T=n_epochs * m,
        batch_size=1,
        beta=beta,
        log_every=m,
        verbose=False,
    )
    x_final = solver.run()
    return sign_invariant_dist(x_final, true_x)


def _worker_l2(data, d, x_init, lr, n_epochs, m, true_x):
    x = x_init.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    a_all = data[:, :-1]
    b_all = data[:, -1]
    for _ in range(n_epochs):
        perm = torch.randperm(m)
        for j in range(m):
            optimizer.zero_grad()
            idx = perm[j]
            loss = (torch.dot(a_all[idx], x) ** 2 - b_all[idx]) ** 2
            loss.backward()
            optimizer.step()
    return sign_invariant_dist(x.detach(), true_x)


def run_outlier_sweep(d, m, ps, beta_invs, lrs, n_epochs, n_rounds, data_seed=42):
    print(f"\n[D] outlier robustness: d={d}, m={m}, p in {ps}")
    clean, true_x = generate_phase_retrieval_data(d, m, seed=data_seed)
    n_workers = max(1, os.cpu_count() - 1)
    n_p, n_b, n_l = len(ps), len(beta_invs), len(lrs)

    # Results indexed explicitly so completion order never matters.
    l1_res = np.full((n_p, n_rounds, n_b), np.nan)
    l2_res = np.full((n_p, n_rounds, n_l), np.nan)

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {}
        for pi, p in enumerate(ps):
            for r in range(n_rounds):
                data = corrupt(clean, p, seed=data_seed + 7 * r + 1)
                x0 = spectral_init(data, d)
                for bi, beta_inv in enumerate(beta_invs):
                    fut = pool.submit(
                        _worker_l1, data, d, x0, 1.0 / beta_inv, n_epochs, m, true_x
                    )
                    futures[fut] = ("l1", pi, r, bi)
                for li, lr in enumerate(lrs):
                    fut = pool.submit(_worker_l2, data, d, x0, lr, n_epochs, m, true_x)
                    futures[fut] = ("l2", pi, r, li)

        for fut in as_completed(futures):
            kind, pi, r, si = futures[fut]
            if kind == "l1":
                l1_res[pi, r, si] = fut.result()
            else:
                l2_res[pi, r, si] = fut.result()

    # Best stepsize per (p, round), then average over rounds.
    l1_err = np.nanmin(l1_res, axis=2).mean(axis=1)
    l2_err = np.nanmin(l2_res, axis=2).mean(axis=1)
    return np.array(ps, dtype=float), l1_err, l2_err


def plot_d(ps, l1_err, l2_err, d, m, save_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ps, l1_err, "C2-o", linewidth=1.6, label="Robust L1 (prox-point)")
    ax.plot(ps, l2_err, "C3--s", linewidth=1.6, label="Smooth L2 (Adam)")
    ax.set_xlabel("Outlier fraction $p$")
    ax.set_ylabel(r"recovery error $\min(\|x-\bar x\|, \|x+\bar x\|)$")
    ax.set_yscale("log")
    ax.set_title(f"Outlier robustness — (d, m) = ({d}, {m})")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"outlier_robustness_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d, m = 50, 300
    ps = [0.0, 0.05, 0.1, 0.2, 0.3]
    beta_invs = np.logspace(-2.5, -0.5, 5)
    lrs = np.logspace(-3, -1, 5)
    n_epochs, n_rounds = 150, 15

    ps_arr, l1_err, l2_err = run_outlier_sweep(
        d, m, ps, beta_invs, lrs, n_epochs, n_rounds
    )
    plot_d(ps_arr, l1_err, l2_err, d, m, save_dir)
    print("\nOutlier robustness experiment complete.")


if __name__ == "__main__":
    main()
