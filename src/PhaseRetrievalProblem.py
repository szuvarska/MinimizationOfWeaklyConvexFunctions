import torch
from src.WeaklyConvexProblem import WeaklyConvexProblem


def f_stochastic(x, batch):
    # |<a, x>^2 - b|
    a, b = batch[:, :-1], batch[:, -1]
    return torch.abs((x @ a.T) ** 2 - b)


class PhaseRetrievalProblem(WeaklyConvexProblem):
    def __init__(self, rho=2.0):
        # We define a dummy model_gen because solve_exact will override it
        super().__init__(f_objective=f_stochastic, r_objective=lambda x: 0,
                         model_gen=None, rho=rho)

    def solve_exact(self, x_t, batch, beta_t):
        """
        Closed-form solution for Stochastic Prox-Linear Phase Retrieval.
        Based on Equation 5.2 in Davis & Drusvyatskiy (2018).
        """
        # batch represents (a_i, b_i)
        # For simplicity, let's assume batch is a single sample (a, b)
        a = batch[0, :-1]
        b = batch[0, -1]

        # Current residual and gradient of the inner quadratic
        dot_prod = torch.dot(a, x_t)
        gamma = dot_prod ** 2 - b
        zeta = 2 * dot_prod * a

        # Step size lambda in the paper is 1/beta_t
        # The formula: Delta = proj_[-1, 1](-gamma / (lambda * ||zeta||^2)) * lambda * zeta
        # Simplified for our beta_t notation:
        norm_zeta_sq = torch.norm(zeta) ** 2

        if norm_zeta_sq < 1e-9:
            return x_t

        scale = torch.clamp(-gamma * beta_t / norm_zeta_sq, min=-1.0, max=1.0)
        delta = (scale / beta_t) * zeta

        return x_t + delta
