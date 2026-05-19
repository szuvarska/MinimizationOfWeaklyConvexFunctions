import torch
from src.WeaklyConvexProblem import WeaklyConvexProblem


def f_stochastic(x, batch):
    # |<a, x>^2 - b|
    a, b = batch[:, :-1], batch[:, -1]
    return torch.abs((x @ a.T) ** 2 - b)


class PhaseRetrievalProblem(WeaklyConvexProblem):
    def __init__(self, rho=2.0):
        # We define a dummy model_gen because solve_exact will override it
        super().__init__(
            f_objective=f_stochastic, r_objective=lambda x: 0, model_gen=None, rho=rho
        )

    def solve_exact(self, x_t, batch, beta_t):
        """
        Closed-form solution for Stochastic Prox-Linear Phase Retrieval.
        Based on Equation 5.2 in Davis & Drusvyatskiy (2018).
        Averages the update over all samples in the batch.
        """
        a_all = batch[:, :-1]  # (batch_size, d)
        b_all = batch[:, -1]  # (batch_size,)

        delta_sum = torch.zeros_like(x_t)
        count = 0

        for i in range(batch.shape[0]):
            a = a_all[i]
            b = b_all[i]

            dot_prod = torch.dot(a, x_t)
            gamma = dot_prod**2 - b
            zeta = 2 * dot_prod * a

            norm_zeta_sq = torch.norm(zeta) ** 2
            if norm_zeta_sq < 1e-9:
                continue

            scale = torch.clamp(-gamma * beta_t / norm_zeta_sq, min=-1.0, max=1.0)
            delta_sum += (scale / beta_t) * zeta
            count += 1

        if count == 0:
            return x_t

        return x_t + delta_sum / count
