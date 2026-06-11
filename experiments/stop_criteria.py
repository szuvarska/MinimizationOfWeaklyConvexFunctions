"""
Experiment F: Stop criteria via the Moreau envelope (TODO section 8.F).

The Moreau gradient norm ||grad phi_lambda(x)|| is the paper's computable
stationarity measure: a small value certifies that x is near a point xhat with
dist(0, d phi(xhat)) <= ||grad phi_lambda(x)|| and phi(xhat) <= phi(x), WITHOUT
knowing phi*. This experiment compares four stopping rules on a phase-retrieval
run (prox-point):

  1. Moreau gradient norm <= eps_m   (principled; no phi* needed)
  2. Function gap <= eps_g           (oracle reference; only because phi* = 0 here)
  3. Iterate stagnation ||x_t - x_{t-1}|| <= eps_s   (cheap)
  4. Fixed budget                    (run to the end)

Curves are averaged over rounds and the stop epochs marked on the convergence
plot. The solver's built-in Moreau early stop (stop_measure="moreau") is also
demonstrated and should agree with rule 1.
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


def generate_phase_retrieval_data(d, m, seed=0):
    torch.manual_seed(seed)
    true_x = torch.randn(d)
    true_x = true_x / torch.norm(true_x)
    a_matrix = torch.randn(m, d)
    b_vector = (a_matrix @ true_x) ** 2
    data = torch.cat([a_matrix, b_vector.unsqueeze(1)], dim=1)
    return data, true_x


def _worker(data, d, beta, n_epochs, m, seed):
    """One trajectory with per-epoch Moreau + step logging."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)
    prob = ProximalPointPhaseRetrieval(rho=2.0)
    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=n_epochs * m,
        batch_size=1,
        beta=beta,
        log_every=m,
        moreau_every=m,
        verbose=False,
    )
    solver.run()
    obj = solver.history["obj_values"][1:]  # drop the init entry
    moreau = [v for _, v in solver.history["moreau_grad_norms"]]
    step = [v for _, v in solver.history["step_norms"]]
    n = min(len(obj), len(moreau), len(step))
    return np.array(obj[:n]), np.array(moreau[:n]), np.array(step[:n])


def run_trajectories(d, m, beta_inv, n_epochs, n_rounds, data_seed=42):
    print(f"\n[F] stop criteria: d={d}, m={m}, beta^-1={beta_inv}")
    data, _ = generate_phase_retrieval_data(d, m, seed=data_seed)
    beta = 1.0 / beta_inv

    n_workers = max(1, os.cpu_count() - 1)
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = [
            pool.submit(_worker, data, d, beta, n_epochs, m, 13 * r + 1)
            for r in range(n_rounds)
        ]
        runs = [f.result() for f in as_completed(futures)]

    n = min(len(r[0]) for r in runs)
    obj = np.mean([r[0][:n] for r in runs], axis=0)
    moreau = np.mean([r[1][:n] for r in runs], axis=0)
    step = np.mean([r[2][:n] for r in runs], axis=0)
    epochs = np.arange(n)
    return epochs, obj, moreau, step, data, beta


def first_below(values, eps):
    idx = np.where(np.asarray(values) <= eps)[0]
    return int(idx[0]) if len(idx) else None


def demo_builtin_stop(data, d, beta, n_epochs, m, eps_m, seed=1):
    """Run the solver's built-in Moreau early stop; return stop epoch (or None)."""
    torch.manual_seed(seed)
    x_init = torch.randn(d)
    x_init = x_init / torch.norm(x_init)
    prob = ProximalPointPhaseRetrieval(rho=2.0)
    solver = ModelBasedSolver(
        problem=prob,
        data=data,
        x_init=x_init,
        T=n_epochs * m,
        batch_size=1,
        beta=beta,
        log_every=m,
        moreau_every=m,
        verbose=False,
        tol=eps_m,
        patience=2,
        stop_measure="moreau",
    )
    solver.run()
    return None if solver.stop_iter is None else solver.stop_iter // m


def plot_f(epochs, obj, moreau, stops, eps, d, m, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = {"Moreau": "C2", "Gap (oracle)": "C0", "Stagnation": "C1", "Budget": "C7"}

    ax1.plot(epochs, np.maximum(obj, 1e-12), "k-", linewidth=1.5, label="function gap")
    for rule, ep in stops.items():
        if ep is not None:
            ax1.axvline(
                ep,
                color=colors[rule],
                linestyle="--",
                alpha=0.8,
                label=f"{rule} stop @ {ep}",
            )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel(r"$\varphi(x_t)$ (gap, $\varphi^*=0$)")
    ax1.set_yscale("log")
    ax1.set_title(f"Stopping rules on the convergence curve — (d,m)=({d},{m})")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, which="both")

    ax2.plot(
        epochs,
        np.maximum(moreau, 1e-12),
        "C2-",
        linewidth=1.5,
        label=r"$\|\nabla\varphi_\lambda(x_t)\|$",
    )
    ax2.axhline(
        eps["Moreau"], color="C2", linestyle=":", alpha=0.7, label=r"$\varepsilon_m$"
    )
    if stops["Moreau"] is not None:
        ax2.axvline(stops["Moreau"], color="C2", linestyle="--", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel(r"$\|\nabla\varphi_\lambda(x_t)\|$")
    ax2.set_yscale("log")
    ax2.set_title("Moreau stationarity measure")
    ax2.legend()
    ax2.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"stop_criteria_d{d}_m{m}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    save_dir = os.path.join(os.path.dirname(__file__), "..", "deliverables", "figures")
    d, m = 50, 150
    beta_inv = 0.1
    n_epochs, n_rounds = 120, 5
    eps = {"Moreau": 0.1, "Gap (oracle)": 1e-3, "Stagnation": 1e-3}

    epochs, obj, moreau, step, data, beta = run_trajectories(
        d, m, beta_inv, n_epochs, n_rounds
    )

    stops = {
        "Moreau": first_below(moreau, eps["Moreau"]),
        "Gap (oracle)": first_below(obj, eps["Gap (oracle)"]),
        "Stagnation": first_below(step, eps["Stagnation"]),
        "Budget": len(epochs) - 1,
    }

    builtin = demo_builtin_stop(data, d, beta, n_epochs, m, eps["Moreau"])

    print("\n  Stopping summary")
    print(f"  {'rule':<16}{'stop epoch':>12}{'gap@stop':>14}{'moreau@stop':>14}")
    for rule, ep in stops.items():
        if ep is None:
            print(f"  {rule:<16}{'never':>12}")
        else:
            print(f"  {rule:<16}{ep:>12}{obj[ep]:>14.2e}{moreau[ep]:>14.2e}")
    print(f"  built-in solver Moreau stop epoch: {builtin}")

    plot_f(epochs, obj, moreau, stops, eps, d, m, save_dir)
    print("\nStop criteria experiment complete.")


if __name__ == "__main__":
    main()
