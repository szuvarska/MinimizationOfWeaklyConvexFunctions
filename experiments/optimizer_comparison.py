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
from concurrent.futures import ProcessPoolExecutor, as_completed

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
        problem=prob,
        data=data,
        x_init=x_init,
        T=T,
        batch_size=1,
        beta=beta,
        log_every=m,
        verbose=False,
    )
    solver.run()
    return (
        solver.history["obj_values"][-1]
        if solver.history["obj_values"]
        else float("inf")
    )


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


MODEL_METHODS = {
    "Subgradient": SubgradientPhaseRetrieval,
    "Prox-Linear": ProxLinearPhaseRetrieval,
    "Proximal Point": ProximalPointPhaseRetrieval,
}

TORCH_OPTS = {
    "SGD": (torch.optim.SGD, {}),
    "Adam": (torch.optim.Adam, {}),
    "AdaGrad": (torch.optim.Adagrad, {}),
}


def _worker_model(method_name, data, d, beta, n_epochs, m, seed):
    """Worker for model-based methods."""
    return run_model_based(MODEL_METHODS[method_name], data, d, beta, n_epochs, m, seed)


def _worker_torch(opt_name, data, d, lr, n_epochs, m, seed):
    """Worker for torch optimizers."""
    opt_cls, kwargs = TORCH_OPTS[opt_name]
    return run_torch_optimizer(opt_cls, data, d, lr, n_epochs, m, seed, **kwargs)


def run_comparison(d, m, n_stepsizes=100, n_epochs=100, n_rounds=15, data_seed=42):
    print(f"\n{'='*60}")
    print(f"  Optimizer Comparison: d={d}, m={m}")
    print(f"  {n_stepsizes} step-sizes, {n_epochs} epochs, {n_rounds} rounds")
    print(f"{'='*60}")

    data, true_x = generate_phase_retrieval_data(d, m, seed=data_seed)

    inv_betas = np.logspace(-4, 0, n_stepsizes)
    lrs = np.logspace(-4, 0, n_stepsizes)

    results = {}
    n_workers = max(1, os.cpu_count() - 1)

    # Build all tasks
    tasks_model = []
    for name in MODEL_METHODS:
        for si, inv_beta in enumerate(inv_betas):
            beta = 1.0 / inv_beta
            for r in range(n_rounds):
                tasks_model.append((si, name, beta, r))

    tasks_torch = []
    for name in TORCH_OPTS:
        for si, lr in enumerate(lrs):
            for r in range(n_rounds):
                tasks_torch.append((si, name, lr, r))

    total = len(tasks_model) + len(tasks_torch)
    print(f"  Running {total} tasks on {n_workers} workers...")

    done = 0

    # Model-based methods
    raw_model = {(si, name): [] for si in range(n_stepsizes) for name in MODEL_METHODS}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker_model, name, data, d, beta, n_epochs, m, r): (si, name)
            for si, name, beta, r in tasks_model
        }
        for future in as_completed(futures):
            si, name = futures[future]
            raw_model[(si, name)].append(future.result())
            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    for name in MODEL_METHODS:
        vals = np.zeros(n_stepsizes)
        for si in range(n_stepsizes):
            vals[si] = np.mean(raw_model[(si, name)])
        results[name] = vals
        print(f"  Done: {name}")

    # Torch optimizers
    raw_torch = {(si, name): [] for si in range(n_stepsizes) for name in TORCH_OPTS}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_worker_torch, name, data, d, lr, n_epochs, m, r): (si, name)
            for si, name, lr, r in tasks_torch
        }
        for future in as_completed(futures):
            si, name = futures[future]
            raw_torch[(si, name)].append(future.result())
            done += 1
            if done % 100 == 0:
                print(f"  Progress: {done}/{total} ({100*done/total:.0f}%)")

    for name in TORCH_OPTS:
        vals = np.zeros(n_stepsizes)
        for si in range(n_stepsizes):
            vals[si] = np.mean(raw_torch[(si, name)])
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

    ax.axhline(
        y=initial_error, color="gray", linestyle=":", alpha=0.5, label="Initial error"
    )
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
