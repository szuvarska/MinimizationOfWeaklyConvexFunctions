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

    def solve_exact(self, x_t, batch, beta_t):
        """
        Optional: Implement the exact argmin for Algorithm 4.1.
        Returns None if no closed-form solution is available.
        """
        return None