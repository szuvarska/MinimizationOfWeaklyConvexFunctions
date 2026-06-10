import torch
from src.WeaklyConvexProblem import WeaklyConvexProblem


def f_robust_regression(x, batch):
    """Per-sample residual |<a, x> - b|."""
    a, b = batch[:, :-1], batch[:, -1]
    return torch.abs((x @ a.T) - b)


class RobustRegressionSubgradient(WeaklyConvexProblem):
    """
    Stochastic subgradient (model 1.4) on robust linear regression

        phi(x) = E |<a, x> - b|  +  (mu/2) ||x||^2.

    f(x) = |<a, x> - b| is convex (absolute value of an affine map), so the
    linear model f_x(y) = f(x) + <v, y - x> is a true under-estimator (tau = 0).
    By Davis & Drusvyatskiy (2019, p.5):
      * mu = 0  -> convex problem, function-value rate O(eps^-2)   (slope -1/2)
      * mu > 0  -> mu-strongly convex, rate O(1/(mu*eps))          (slope -1)

    Subgradient: d/dx |<a, x> - b| = sign(<a, x> - b) * a.
    Closed-form update with r(x) = (mu/2)||x||^2:
        x+ = argmin_y <v, y - x> + (mu/2)||y||^2 + (beta/2)||y - x||^2
           = (beta * x - v) / (beta + mu).
    """

    def __init__(self, rho=0.0, mu=0.0):
        self.mu = mu
        super().__init__(
            f_objective=f_robust_regression,
            r_objective=lambda x: 0.5 * mu * torch.sum(x**2),
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, x_t, batch, beta_t):
        a_all = batch[:, :-1]
        b_all = batch[:, -1]

        grad_sum = torch.zeros_like(x_t)
        for i in range(batch.shape[0]):
            a = a_all[i]
            b = b_all[i]
            residual = torch.dot(a, x_t) - b
            grad_sum += torch.sign(residual) * a
        v = grad_sum / batch.shape[0]

        return (beta_t * x_t - v) / (beta_t + self.mu)
