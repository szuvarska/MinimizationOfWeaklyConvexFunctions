import torch
from src.WeaklyConvexProblem import WeaklyConvexProblem
from src.regularizers import l1_prox


def f_phase_retrieval(x, batch):
    """g(x) = |<a, x>^2 - b| averaged over the batch."""
    a, b = batch[:, :-1], batch[:, -1]
    return torch.abs((x @ a.T) ** 2 - b)


class SparsePhaseRetrievalSubgradient(WeaklyConvexProblem):
    """
    Proximal stochastic subgradient (model 1.4) for sparse phase retrieval:

        phi(x) = E |<a, x>^2 - b|  +  mu * ||x||_1.

    This is the r != 0 instance of the paper's framework phi = f + r. The
    update is the proximal subgradient step

        x+ = prox_{(mu/beta) ||.||_1}( x - (1/beta) v ),   v in d f(x),

    where the L1 prox is soft-thresholding (src.regularizers.l1_prox). With
    mu = 0 this reduces to the plain subgradient method, used as the
    "no regularization" baseline.

    Subdifferential of g(x) = |<a, x>^2 - b|:
        d g(x) = 2 <a, x> a * sign(<a, x>^2 - b).
    """

    def __init__(self, rho=2.0, mu=0.0):
        self.mu = mu
        super().__init__(
            f_objective=f_phase_retrieval,
            r_objective=lambda x: mu * torch.norm(x, p=1),
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, x_t, batch, beta_t):
        a_all = batch[:, :-1]
        b_all = batch[:, -1]
        lam = 1.0 / beta_t

        grad_sum = torch.zeros_like(x_t)
        for i in range(batch.shape[0]):
            a = a_all[i]
            b = b_all[i]
            dot_prod = torch.dot(a, x_t)
            residual = dot_prod**2 - b
            sign = torch.sign(residual) if residual.abs() > 1e-12 else torch.tensor(0.0)
            grad_sum += 2 * dot_prod * sign * a
        v = grad_sum / batch.shape[0]

        z = x_t - lam * v
        if self.mu > 0:
            return l1_prox(z, lam * self.mu)
        return z
