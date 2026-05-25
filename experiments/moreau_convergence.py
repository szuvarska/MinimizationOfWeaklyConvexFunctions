"""
Experiment 4: Moreau Envelope Gradient Convergence

Tracks ||∇φ_λ(x_t)|| over epochs for each of the three stochastic
model-based methods on the phase retrieval problem.

This demonstrates the paper's main convergence guarantee: the Moreau
envelope gradient norm tends to zero, confirming that iterates approach
near-stationarity (Theorems 4.2, 4.5, 4.8).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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


def run_single(prob, data, d, beta, n_epochs, m, moreau_epoch_interval, seed):
    """Run one trial, tracking Moreau gradient norm at regular epoch intervals."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)

    T = n_epochs * m
    log_every = m  # log once per epoch
    moreau_every = moreau_epoch_interval * m  # compute Moreau every N epochs

    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=T,
        batch_size=1,
        beta=beta,
        log_every=log_every,
        moreau_every=moreau_every,
        verbose=False,
    )
    solver.run()

    return solver.history


def run_moreau_experiment(
    d=50,
    m=150,
    beta_inv=0.1,
    n_epochs=100,
    moreau_epoch_interval=5,
    n_rounds=5,
    data_seed=42,
):
    """Run all 3 methods, averaging Moreau gradient norms over rounds."""
    print(f"\n{'='*60}")
    print(f"  Moreau Envelope Convergence: d={d}, m={m}, β⁻¹={beta_inv}")
    print(
        f"  {n_epochs} epochs, Moreau every {moreau_epoch_interval} epochs, {n_rounds} rounds"
    )
    print(f"{'='*60}")

    data, true_x = generate_phase_retrieval_data(d, m, seed=data_seed)
    beta = 1.0 / beta_inv

    method_classes = {
        "Subgradient": SubgradientPhaseRetrieval,
        "Prox-Linear": ProxLinearPhaseRetrieval,
        "Proximal Point": ProximalPointPhaseRetrieval,
    }

    results = {}

    for name, cls in method_classes.items():
        prob = cls(rho=2.0)
        all_moreau = []
        all_obj = []

        for r in range(n_rounds):
            history = run_single(
                prob,
                data,
                d,
                beta,
                n_epochs,
                m,
                moreau_epoch_interval,
                seed=r,
            )
            # moreau_grad_norms is list of (iteration, value) tuples
            moreau_pairs = history["moreau_grad_norms"]
            moreau_epochs = [it / m for it, _ in moreau_pairs]
            moreau_vals = [val for _, val in moreau_pairs]
            all_moreau.append(moreau_vals)

            all_obj.append(history["obj_values"])
            print(f"  {name} round {r+1}/{n_rounds} done")

        # Average over rounds (trim to shortest)
        min_len_m = min(len(x) for x in all_moreau)
        avg_moreau = np.mean([x[:min_len_m] for x in all_moreau], axis=0)

        min_len_o = min(len(x) for x in all_obj)
        avg_obj = np.mean([x[:min_len_o] for x in all_obj], axis=0)

        results[name] = {
            "moreau_epochs": moreau_epochs[:min_len_m],
            "moreau_vals": avg_moreau,
            "obj_epochs": list(range(len(avg_obj))),
            "obj_vals": avg_obj,
        }

    return results


def plot_moreau_convergence(results, d, m, beta_inv, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Moreau envelope gradient norm
    for name, data in results.items():
        ax1.plot(
            data["moreau_epochs"],
            data["moreau_vals"],
            label=name,
            linewidth=1.5,
            marker="o",
            markersize=3,
        )

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$\|\nabla \varphi_\lambda(x_t)\|$")
    ax1.set_title(
        f"Moreau envelope gradient — (d, m) = ({d}, {m}), "
        r"$\beta^{-1}$" + f" = {beta_inv}"
    )
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Objective value
    for name, data in results.items():
        ax2.plot(data["obj_epochs"], data["obj_vals"], label=name, linewidth=1.5)

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$\varphi(x_t)$")
    ax2.set_title(
        f"Objective value — (d, m) = ({d}, {m}), " r"$\beta^{-1}$" + f" = {beta_inv}"
    )
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"moreau_convergence_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")

    configs = [
        {"d": 10, "m": 30, "beta_inv": 0.1},
        {"d": 50, "m": 150, "beta_inv": 0.1},
        {"d": 100, "m": 300, "beta_inv": 0.1},
    ]

    for cfg in configs:
        results = run_moreau_experiment(
            d=cfg["d"],
            m=cfg["m"],
            beta_inv=cfg["beta_inv"],
            n_epochs=100,
            moreau_epoch_interval=5,
            n_rounds=5,
        )
        plot_moreau_convergence(results, cfg["d"], cfg["m"], cfg["beta_inv"], save_dir)

    print("\nAll Moreau convergence experiments complete.")


if __name__ == "__main__":
    main()
