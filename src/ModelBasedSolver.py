import torch
import numpy as np


class ModelBasedSolver:
    """
    Implements Algorithm 4.1: Stochastic Model-Based Minimization.

    From Davis & Drusvyatskiy (2019):
        x_{t+1} = argmin { r(x) + f_{x_t}(x, ξ_t) + (β_t/2)||x - x_t||² }

    The paper requires β_t > ρ̄ > τ + η. For simplicity and following the
    paper's experiments (Section 5), we use a constant β across iterations.
    """

    def __init__(
        self,
        problem,
        data,
        x_init,
        T,
        batch_size=1,
        beta=None,
        log_every=50,
        moreau_every=None,
        verbose=True,
        true_solution=None,
    ):
        self.prob = problem
        self.data = data
        self.x = x_init.clone().detach()
        self.T = T
        self.batch_size = batch_size
        self.log_every = log_every
        self.moreau_every = moreau_every
        self.verbose = verbose
        self.true_solution = true_solution

        # β must satisfy β > ρ (paper: β > ρ̄ > τ+η).
        # Default: β = 2ρ + 1, safely above ρ.
        if beta is None:
            self.beta = 2.0 * self.prob.rho + 1.0
        else:
            self.beta = beta

        # History for convergence tracking
        self.history = {
            "iterations": [],
            "obj_values": [],
            "moreau_grad_norms": [],
            "x_norms": [],
            "dist_to_solution": [],
        }

    def run(self):
        # Log the initial point before any updates
        obj_val = self.prob.population_objective(self.x, self.data)
        x_norm = torch.norm(self.x).item()
        self.history["iterations"].append(-1)
        self.history["obj_values"].append(obj_val)
        self.history["x_norms"].append(x_norm)
        if self.true_solution is not None:
            dist = torch.norm(self.x - self.true_solution).item()
            self.history["dist_to_solution"].append(dist)
        if self.verbose:
            print(f"Iter init: ||x|| = {x_norm:.4f} | phi(x) = {obj_val:.6f}")

        for t in range(self.T):
            beta_t = self.beta

            indices = torch.randint(0, len(self.data), (self.batch_size,))
            batch = self.data[indices]

            # Check for closed-form solution first
            exact_x = self.prob.solve_exact(self.x, batch, beta_t)

            if exact_x is not None:
                self.x = exact_x.detach().clone()
            else:
                # Fallback to LBFGS numerical solver
                x_t = self.x.clone().detach()
                g_xt = self.prob.get_model(x_t, batch)

                y = x_t.clone().requires_grad_(True)
                inner_opt = torch.optim.LBFGS([y], lr=1, max_iter=20)

                def closure():
                    inner_opt.zero_grad()
                    model_val = g_xt(y) + self.prob.r(y)
                    penalty = (beta_t / 2) * torch.norm(y - x_t) ** 2
                    loss = model_val + penalty
                    loss.backward()
                    return loss

                inner_opt.step(closure)
                self.x = y.detach().clone()

            # Logging
            if t % self.log_every == 0:
                obj_val = self.prob.population_objective(self.x, self.data)
                x_norm = torch.norm(self.x).item()

                self.history["iterations"].append(t)
                self.history["obj_values"].append(obj_val)
                self.history["x_norms"].append(x_norm)

                if self.true_solution is not None:
                    dist = torch.norm(self.x - self.true_solution).item()
                    self.history["dist_to_solution"].append(dist)

                # Moreau envelope gradient norm (expensive, compute less often)
                if self.moreau_every and t % self.moreau_every == 0:
                    moreau_gn = self.prob.compute_moreau_grad_norm(self.x, self.data)
                    self.history["moreau_grad_norms"].append((t, moreau_gn))

                if self.verbose:
                    print(
                        f"Iter {t:4d}: ||x|| = {x_norm:.4f} | "
                        f"phi(x) = {obj_val:.6f}"
                    )

        return self.x
