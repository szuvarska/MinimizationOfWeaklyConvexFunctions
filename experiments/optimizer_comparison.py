"""
Experiment 3: Compare model-based methods with standard optimizers (SGD, Adam, AdaGrad)
on the phase retrieval problem.

Shows step-size sensitivity advantage of prox-linear / proximal point over SGD/Adam/AdaGrad.
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
    f_phase_retrieval,
)


def generate_phase_retrieval_data(d, m, seed=0):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    population_data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return population_data, true_x


def run_model_based(method_cls, data, d, beta, n_epochs, m, seed):
    """Run one model-based trial. Returns final population objective."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)
    T = n_epochs * m
    prob = method_cls(rho=2.0)
    solver = ModelBasedSolver(
        problem=prob, data=data, x_init=x_init,
        T=T, batch_size=1, beta=beta, log_every=m, verbose=False,
    )
    solver.run()
    return solver.history["obj_values"][-1] if solver.history["obj_values"] else float("inf")


def run_torch_optimizer(opt_class, data, d, lr, n_epochs, m, seed, **opt_kwargs):
    """Run a torch optimizer on the phase retrieval loss. Returns final population objective."""
    torch.manual_seed(seed)
    x = torch.randn(d)
    x = x / torch.norm(x)
    x = x.clone().requires_grad_(True)

    optimizer = opt_class([x], lr=lr, **opt_kwargs)

    a_all = data[:, :-1]
    b_all = data[:, -1]

    for epoch in range(n_epochs):
        perm = torch.randperm(m)
        for j in range(m):
            optimizer.zero_grad()
            idx = perm[j]
            a = a_all[idx]
            b = b_all[idx]
            loss = torch.abs(torch.dot(a, x) ** 2 - b)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        obj = torch.mean(torch.abs((a_all @ x) ** 2 - b_all)).item()
    return obj


def run_comparison(d, m, n_stepsizes=100, n_epochs=100, n_rounds=15, data_seed=42):
    print(f"\n{'='*60}")
    print(f"  Optimizer Comparison: d={d}, m={m}")
    print(f"  {n_stepsizes} step-sizes, {n_epochs} epochs, {n_rounds} rounds")
    print(f"{'='*60}")

    data, true_x = generate_phase_retrieval_data(d, m, seed=data_seed)

    # Model-based methods use inv_beta as step-size parameter
    inv_betas = np.logspace(-4, 0, n_stepsizes)

    # Torch optimizers use learning rate
    lrs = np.logspace(-4, 0, n_stepsizes)

    # Model-based methods
    model_methods = {
        "Subgradient": SubgradientPhaseRetrieval,
        "Prox-Linear": ProxLinearPhaseRetrieval,
        "Proximal Point": ProximalPointPhaseRetrieval,
    }

    # Torch optimizers
    torch_opts = {
        "SGD": (torch.optim.SGD, {}),
        "Adam": (torch.optim.Adam, {}),
        "AdaGrad": (torch.optim.Adagrad, {}),
    }

    results = {}

    # Run model-based methods
    for name, cls in model_methods.items():
        vals = np.zeros(n_stepsizes)
        for si, inv_beta in enumerate(inv_betas):
            beta = 1.0 / inv_beta
            objs = []
            for r in range(n_rounds):
                obj = run_model_based(cls, data, d, beta, n_epochs, m, seed=r)
                objs.append(obj)
            vals[si] = np.mean(objs)
        results[name] = vals
        print(f"  Done: {name}")

    # Run torch optimizers
    for name, (opt_cls, kwargs) in torch_opts.items():
        vals = np.zeros(n_stepsizes)
        for si, lr in enumerate(lrs):
            objs = []
            for r in range(n_rounds):
                obj = run_torch_optimizer(opt_cls, data, d, lr, n_epochs, m, seed=r, **kwargs)
                objs.append(obj)
            vals[si] = np.mean(objs)
        results[name] = vals
        print(f"  Done: {name}")

    return lrs, results


def plot_comparison(d, m, lrs, results, initial_error, save_dir):
    fig, ax = plt.subplots(figsize=(10, 6))

    model_names = ["Subgradient", "Prox-Linear", "Proximal Point"]
    opt_names = ["SGD", "Adam", "AdaGrad"]
    styles = {
        "Subgradient": {"color": "C0", "linestyle": "-"},
        "Prox-Linear": {"color": "C1", "linestyle": "-"},
        "Proximal Point": {"color": "C2", "linestyle": "-"},
        "SGD": {"color": "C3", "linestyle": "--"},
        "Adam": {"color": "C4", "linestyle": "--"},
        "AdaGrad": {"color": "C5", "linestyle": "--"},
    }

    for name in model_names + opt_names:
        ax.plot(lrs, results[name], label=name, linewidth=1.5, **styles[name])

    ax.axhline(y=initial_error, color="gray", linestyle=":", alpha=0.5, label="Initial error")
    ax.set_xlabel("Step-size parameter")
    ax.set_ylabel("Function value after 100 epochs")
    ax.set_title(f"Model-based vs Standard Optimizers — (d, m) = ({d}, {m})")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"optimizer_comparison_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    configs = [(10, 30), (50, 150), (100, 300)]
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")

    for d, m in configs:
        lrs, results = run_comparison(d, m)

        data, _ = generate_phase_retrieval_data(d, m, seed=42)
        prob = SubgradientPhaseRetrieval(rho=2.0)
        torch.manual_seed(0)
        x0 = torch.randn(d)
        x0 = x0 / torch.norm(x0)
        initial_error = prob.population_objective(x0, data)

        plot_comparison(d, m, lrs, results, initial_error, save_dir)

    print("\nAll optimizer comparison experiments complete.")


if __name__ == "__main__":
    main()
