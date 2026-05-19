import torch


class WeaklyConvexProblem:
    """
    Defines the objective function phi(x) = f(x) + r(x).
    Based on Example 2.1 (Phase Retrieval) or simple |x^2 - 1|.
    f_objective: The stochastic loss function f(x)
    """

    def __init__(self, f_objective, r_objective, model_gen, rho=2.0):
        self.f = f_objective
        self.r = r_objective
        self.get_model = model_gen
        self.rho = rho  # Used to guide the Solver's beta_t

    def objective(self, x, data_batch=None):
        return self.f(x, data_batch) + self.r(x)

    def population_objective(self, x, data):
        with torch.no_grad():
            mean_f = torch.mean(self.f(x, data))
            reg_val = self.r(x)
            return (mean_f + reg_val).item()

    def compute_moreau_grad_norm(self, x, data, lam=None, max_iter=50):
        """
        Compute ||∇φ_λ(x)|| = (1/λ)||x - prox_{λφ}(x)||.

        The Moreau envelope is φ_λ(x) = min_y { φ(y) + 1/(2λ)||y-x||² }.
        Its gradient is ∇φ_λ(x) = (1/λ)(x - prox_{λφ}(x)).
        Requires λ < 1/ρ for smoothness (Lemma 2.2 in the paper).

        We approximate prox_{λφ}(x) numerically via L-BFGS.
        """
        if lam is None:
            lam = 1.0 / (2.0 * self.rho)

        x_anchor = x.clone().detach()
        y = x_anchor.clone().requires_grad_(True)
        opt = torch.optim.LBFGS([y], lr=1, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            phi_y = torch.mean(self.f(y, data)) + self.r(y)
            prox_penalty = (1.0 / (2.0 * lam)) * torch.norm(y - x_anchor) ** 2
            loss = phi_y + prox_penalty
            loss.backward()
            return loss

        opt.step(closure)

        with torch.no_grad():
            prox_x = y.detach()
            grad_norm = (1.0 / lam) * torch.norm(x_anchor - prox_x).item()

        return grad_norm

    def solve_exact(self, x_t, batch, beta_t):
        """
        Optional: Implement the exact argmin for Algorithm 4.1.
        Returns None if no closed-form solution is available.
        """
        return None
