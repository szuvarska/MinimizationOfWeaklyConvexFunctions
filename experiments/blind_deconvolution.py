"""
Experiment 2: Blind Deconvolution (Section 5.2)

Reproduces Figure 4 from Davis & Drusvyatskiy (2019).
Three configs: (d1,d2,m) = (10,10,30), (50,50,200), (100,100,400).
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
from src.problems.blind_deconvolution import (
    BlindDeconvSubgradient,
    BlindDeconvProxLinear,
    BlindDeconvProximalPoint,
)


def generate_blind_deconv_data(d1, d2, m, seed=0):
    """Generate blind deconvolution data.

    Returns (data, true_z) where data has rows [u_i, v_i, b_i]
    and true_z = [x̄; ȳ] is on the unit sphere in R^{d1+d2}.
    """
    torch.manual_seed(seed)
    # Target signal on unit sphere in R^{d1+d2}
    true_z = torch.randn(d1 + d2)
    true_z = true_z / torch.norm(true_z)
    true_x = true_z[:d1]
    true_y = true_z[d1:]

    u_matrix = torch.randn(m, d1)
    v_matrix = torch.randn(m, d2)
    b_vector = (u_matrix @ true_x) * (v_matrix @ true_y)

    data = torch.cat([u_matrix, v_matrix, b_vector.unsqueeze(1)], dim=1)
    return data, true_z


def run_single(prob, data, d_total, beta, n_epochs, m, seed):
    """Run one trial. Returns (final_obj, epoch_to_target or None)."""
    torch.manual_seed(seed)
    z_init = torch.randn(d_total)
    z_init = z_init / torch.norm(z_init)

    T = n_epochs * m
    target = 1e-4

    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=z_init,
        T=T,
        batch_size=1,
        beta=beta,
        log_every=m,
        verbose=False,
    )
    solver.run()

    final_obj = (
        solver.history["obj_values"][-1]
        if solver.history["obj_values"]
        else float("inf")
    )

    epoch_to_target = None
    for idx, obj in enumerate(solver.history["obj_values"]):
        if obj < target:
            epoch_to_target = idx + 1
            break

    return final_obj, epoch_to_target


def run_config(d1, d2, m, n_stepsizes=100, n_epochs=100, n_rounds=15, data_seed=42):
    print(f"\n{'='*60}")
    print(f"  Blind Deconvolution: d1={d1}, d2={d2}, m={m}")
    print(f"  {n_stepsizes} step-sizes, {n_epochs} epochs, {n_rounds} rounds")
    print(f"{'='*60}")

    data, true_z = generate_blind_deconv_data(d1, d2, m, seed=data_seed)
    d_total = d1 + d2

    inv_betas = np.logspace(-4, 0, n_stepsizes)
    beta_values = 1.0 / inv_betas

    method_classes = {
        "Subgradient": BlindDeconvSubgradient,
        "Prox-Linear": BlindDeconvProxLinear,
        "Proximal Point": BlindDeconvProximalPoint,
    }

    final_objs = {name: np.zeros(n_stepsizes) for name in method_classes}
    epochs_to_target = {name: np.full(n_stepsizes, np.nan) for name in method_classes}

    total = n_stepsizes * len(method_classes)
    count = 0

    for si, beta in enumerate(beta_values):
        for name, cls in method_classes.items():
            prob = cls(d1=d1, rho=2.0)
            objs = []
            epochs = []

            for r in range(n_rounds):
                obj, epoch = run_single(prob, data, d_total, beta, n_epochs, m, seed=r)
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


def plot_config(
    d1, d2, m, inv_betas, final_objs, epochs_to_target, initial_error, save_dir
):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for name, objs in final_objs.items():
        ax1.plot(inv_betas, objs, label=name, linewidth=1.5)

    ax1.axhline(
        y=initial_error, color="blue", linestyle="--", alpha=0.5, label="Initial error"
    )
    ax1.set_xlabel(r"$\beta^{-1}$")
    ax1.set_ylabel("Function value after 100 epochs")
    ax1.set_title(f"$(d_1, d_2, m) = ({d1}, {d2}, {m})$")
    ax1.set_yscale("log")
    ax1.set_xscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for name in ["Prox-Linear", "Proximal Point"]:
        vals = epochs_to_target[name]
        mask = ~np.isnan(vals)
        if mask.any():
            ax2.plot(inv_betas[mask], vals[mask], label=name, linewidth=1.5)

    ax2.set_xlabel(r"$\beta^{-1}$")
    ax2.set_ylabel(r"Epochs to $10^{-4}$ suboptimality")
    ax2.set_title(f"$(d_1, d_2, m) = ({d1}, {d2}, {m})$")
    ax2.set_xscale("log")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"blind_deconv_d{d1}_{d2}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    configs = [(10, 10, 30), (50, 50, 200), (100, 100, 400)]
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")

    for d1, d2, m in configs:
        inv_betas, final_objs, epochs_to_target = run_config(d1, d2, m)

        data, _ = generate_blind_deconv_data(d1, d2, m, seed=42)
        prob = BlindDeconvSubgradient(d1=d1, rho=2.0)
        d_total = d1 + d2
        torch.manual_seed(0)
        z0 = torch.randn(d_total)
        z0 = z0 / torch.norm(z0)
        initial_error = prob.population_objective(z0, data)

        plot_config(
            d1, d2, m, inv_betas, final_objs, epochs_to_target, initial_error, save_dir
        )

    print("\nAll blind deconvolution experiments complete.")


if __name__ == "__main__":
    main()
