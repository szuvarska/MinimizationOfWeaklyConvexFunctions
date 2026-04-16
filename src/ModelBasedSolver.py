import torch
import numpy as np


class ModelBasedSolver:
    """
    Implements Algorithm 1.2: Stochastic Model-Based Minimization.
    """

    def __init__(self, problem, data, x_init, T, batch_size=1, gamma=0.1):
        self.prob = problem
        self.data = data  # The population P
        self.x = x_init.clone().detach()
        self.T = T
        self.gamma = gamma  # Step-size parameter
        self.batch_size = batch_size
        self.history = []

    def get_beta(self, t):
        # The paper requires beta_t > rho
        # We add a small constant (e.g., 0.1) to ensure strict strong convexity
        rho_hat = self.prob.rho + 0.1
        return rho_hat + (1.0 / self.gamma) * np.sqrt(t + 1.0)

    def run(self):

        for t in range(self.T):
            beta_t = self.get_beta(t)

            indices = torch.randint(0, len(self.data), (self.batch_size,))
            batch = self.data[indices]

            # Check for closed-form solution first
            exact_x = self.prob.solve_exact(self.x, batch, beta_t)

            if exact_x is not None:
                self.x = exact_x.detach().clone()
            else:
                # Fallback to your LBFGS numerical solver
                x_t = self.x.clone().detach()
                g_xt = self.prob.get_model(x_t, batch)

                # SUBPROBLEM: x_{t+1} = argmin { f_xt(y) + (1/2*alpha) * ||y-x_t||^2 }
                y = x_t.clone().requires_grad_(True)
                inner_opt = torch.optim.LBFGS([y], lr=1, max_iter=20)

                def closure():
                    inner_opt.zero_grad()
                    model_val = g_xt(y) + self.prob.r(y)  # Add the regularizer r(y)
                    penalty = (beta_t / 2) * torch.norm(y - x_t) ** 2
                    loss = model_val + penalty
                    loss.backward()
                    return loss

                inner_opt.step(closure)
                self.x = y.detach().clone()

            if t % 50 == 0:
                # Calculate the True Objective (over the whole population)
                # This represents the 'Population Risk'
                with torch.no_grad():
                    mean_f = torch.mean(self.prob.f(self.x, self.data))
                    reg_val = self.prob.r(self.x)
                    total_phi = mean_f + reg_val

                x_norm = torch.norm(self.x).item()
                print(f"Iter {t:4}: x = {x_norm:.4f} | True phi(x) = {total_phi.item():.4f}")

        return self.x