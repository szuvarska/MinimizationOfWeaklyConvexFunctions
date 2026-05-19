"""
Experiment 1: Phase Retrieval (Section 5.1)

Reproduces Figure 3 from Davis & Drusvyatskiy (2019).
Three configs: (d,m) = (10,30), (50,150), (100,300).
100 equally spaced step-sizes β⁻¹ ∈ [10⁻⁴, 1].
100 passes through data, averaged over 15 rounds.

Two plots per config:
  Left:  Function gap after 100 epochs vs β⁻¹ (all 3 methods)
  Right: Epochs to reach 10⁻⁴ suboptimality vs β⁻¹ (prox-linear & prox-point)
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


def run_single(prob, data, d, beta, n_epochs, m, seed):
    """Run one trial. Returns (final_obj, epoch_to_target or None)."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)

    T = n_epochs * m
    target = 1e-4

    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=T,
        batch_size=1,
        beta=beta,
        log_every=m,  # log once per epoch
        verbose=False,
    )
    solver.run()

    final_obj = solver.history["obj_values"][-1] if solver.history["obj_values"] else float("inf")

    # Find first epoch where objective < target
    epoch_to_target = None
    for idx, obj in enumerate(solver.history["obj_values"]):
        if obj < target:
            epoch_to_target = idx + 1  # 1-indexed epoch
            break

    return final_obj, epoch_to_target


def run_config(d, m, n_stepsizes=100, n_epochs=100, n_rounds=15, data_seed=42):
    print(f"\n{'='*60}")
    print(f"  Phase Retrieval: d={d}, m={m}")
    print(f"  {n_stepsizes} step-sizes, {n_epochs} epochs, {n_rounds} rounds")
    print(f"{'='*60}")

    data, true_x = generate_phase_retrieval_data(d, m, seed=data_seed)

    inv_betas = np.linspace(1e-4, 1.0, n_stepsizes)
    beta_values = 1.0 / inv_betas

    method_classes = {
        "Subgradient": SubgradientPhaseRetrieval,
        "Prox-Linear": ProxLinearPhaseRetrieval,
        "Proximal Point": ProximalPointPhaseRetrieval,
    }

    # Results: final_obj[method][step_idx] = avg objective
    # epochs_to_target[method][step_idx] = avg epochs (or None)
    final_objs = {name: np.zeros(n_stepsizes) for name in method_classes}
    epochs_to_target = {name: np.full(n_stepsizes, np.nan) for name in method_classes}

    total = n_stepsizes * len(method_classes)
    count = 0

    for si, beta in enumerate(beta_values):
        for name, cls in method_classes.items():
            prob = cls(rho=2.0)
            objs = []
            epochs = []

            for r in range(n_rounds):
                obj, epoch = run_single(prob, data, d, beta, n_epochs, m, seed=r)
                objs.append(obj)
                if epoch is not None:
                    epochs.append(epoch)

            final_objs[name][si] = np.mean(objs)
            if epochs:
                epochs_to_target[name][si] = np.mean(epochs)

            count += 1
            if count % 10 == 0:
                print(f"  Progress: {count}/{total} ({100*count/total:.0f}%)")

    return inv_betas, final_objs, epochs_to_target


def plot_config(d, m, inv_betas, final_objs, epochs_to_target, initial_error, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: function gap after 100 epochs
    for name, objs in final_objs.items():
        ax1.plot(inv_betas, objs, label=name, linewidth=1.5)

    ax1.axhline(y=initial_error, color="blue", linestyle="--", alpha=0.5, label="Initial error")
    ax1.set_xlabel(r"$\beta^{-1}$")
    ax1.set_ylabel("Function value after 100 epochs")
    ax1.set_title(f"(d, m) = ({d}, {m})")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: epochs to reach 10⁻⁴ (prox-linear & prox-point only)
    for name in ["Prox-Linear", "Proximal Point"]:
        vals = epochs_to_target[name]
        mask = ~np.isnan(vals)
        if mask.any():
            ax2.plot(inv_betas[mask], vals[mask], label=name, linewidth=1.5)

    ax2.set_xlabel(r"$\beta^{-1}$")
    ax2.set_ylabel(r"Epochs to $10^{-4}$ suboptimality")
    ax2.set_title(f"(d, m) = ({d}, {m})")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"phase_retrieval_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    configs = [(10, 30), (50, 150), (100, 300)]
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")

    for d, m in configs:
        inv_betas, final_objs, epochs_to_target = run_config(d, m)

        # Compute initial error for the dashed line
        data, _ = generate_phase_retrieval_data(d, m, seed=42)
        prob = SubgradientPhaseRetrieval(rho=2.0)
        torch.manual_seed(0)
        x0 = torch.randn(d)
        x0 = x0 / torch.norm(x0)
        initial_error = prob.population_objective(x0, data)

        plot_config(d, m, inv_betas, final_objs, epochs_to_target, initial_error, save_dir)

    print("\nAll phase retrieval experiments complete.")


if __name__ == "__main__":
    main()
