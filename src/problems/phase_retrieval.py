import torch
from src.WeaklyConvexProblem import WeaklyConvexProblem


def f_phase_retrieval(x, batch):
    """g(x) = |<a, x>^2 - b| averaged over the batch."""
    a, b = batch[:, :-1], batch[:, -1]
    return torch.abs((x @ a.T) ** 2 - b)


class SubgradientPhaseRetrieval(WeaklyConvexProblem):
    """
    Stochastic subgradient method for phase retrieval (Section 5.1).

    Model (1.4): f_x(y,ξ) = f(x,ξ) + <v, y-x>  where v ∈ ∂f(x,ξ).
    Subdifferential: ∂g(x) = 2<a,x>a · sign(<a,x>²-b).
    Closed-form update: x_{t+1} = x_t - v/β.
    """

    def __init__(self, rho=2.0):
        super().__init__(
            f_objective=f_phase_retrieval,
            r_objective=lambda x: 0,
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
            residual = dot_prod ** 2 - b
            sign = torch.sign(residual) if residual.abs() > 1e-12 else torch.tensor(0.0)
            grad_sum += 2 * dot_prod * sign * a

        v = grad_sum / batch.shape[0]
        return x_t - lam * v


class ProxLinearPhaseRetrieval(WeaklyConvexProblem):
    """
    Stochastic prox-linear method for phase retrieval (Section 5.1).

    Model (1.5): f_x(y,ξ) = h(c(x,ξ) + ∇c(x,ξ)(y-x), ξ)
    For g(x) = |<a,x>²-b|, this linearizes c(x)=<a,x>² inside | · |.
    Closed-form from eq. (5.2): Δ* = proj_{[-1,1]}(-γ/||ζ||²) · ζ
    where γ = (<a,x>²-b)/β, ζ = 2<a,x>a/β.
    """

    def __init__(self, rho=2.0):
        super().__init__(
            f_objective=f_phase_retrieval,
            r_objective=lambda x: 0,
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, x_t, batch, beta_t):
        a_all = batch[:, :-1]
        b_all = batch[:, -1]

        delta_sum = torch.zeros_like(x_t)
        count = 0

        for i in range(batch.shape[0]):
            a = a_all[i]
            b = b_all[i]

            dot_prod = torch.dot(a, x_t)
            gamma = dot_prod ** 2 - b
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


class ProximalPointPhaseRetrieval(WeaklyConvexProblem):
    """
    Stochastic proximal point method for phase retrieval (Section 5.1).

    Model (1.6): f_x(y,ξ) = f(y,ξ)  (the full function itself).
    Subproblem: argmin_y |<a,y>²-b| + (β/2)||y-x||²

    From eq. (5.4), there are at most 4 candidate solutions:
      y = x - (2λ<a,x>/(2λ||a||²+1))a     [sign = +1]
      y = x - (2λ<a,x>/(2λ||a||²-1))a     [sign = -1]
      y = x - ((<a,x>+√b)/||a||²)a         [<a,y>² = b, positive]
      y = x - ((<a,x>-√b)/||a||²)a         [<a,y>² = b, negative]
    where λ = 1/β. Pick the candidate with the lowest subproblem value.
    """

    def __init__(self, rho=2.0):
        super().__init__(
            f_objective=f_phase_retrieval,
            r_objective=lambda x: 0,
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, x_t, batch, beta_t):
        a_all = batch[:, :-1]
        b_all = batch[:, -1]
        lam = 1.0 / beta_t

        x_new = x_t.clone()
        for i in range(batch.shape[0]):
            a = a_all[i]
            b = b_all[i]
            x_new = self._solve_single(x_new, a, b, lam, beta_t)

        return x_new

    def _solve_single(self, x, a, b, lam, beta_t):
        dot_ax = torch.dot(a, x)
        norm_a_sq = torch.dot(a, a)

        candidates = []

        # Candidate 1: sign = +1 → y = x - (2λ<a,x>/(2λ||a||²+1)) a
        denom1 = 2 * lam * norm_a_sq + 1
        y1 = x - (2 * lam * dot_ax / denom1) * a
        candidates.append(y1)

        # Candidate 2: sign = -1 → y = x - (2λ<a,x>/(2λ||a||²-1)) a
        denom2 = 2 * lam * norm_a_sq - 1
        if abs(denom2.item()) > 1e-9:
            y2 = x - (2 * lam * dot_ax / denom2) * a
            candidates.append(y2)

        # Candidate 3: <a,y> = +√b → y = x - ((<a,x>-√b)/||a||²) a
        if b >= 0:
            sqrt_b = torch.sqrt(b)
            y3 = x - ((dot_ax - sqrt_b) / norm_a_sq) * a
            candidates.append(y3)

            # Candidate 4: <a,y> = -√b → y = x - ((<a,x>+√b)/||a||²) a
            y4 = x - ((dot_ax + sqrt_b) / norm_a_sq) * a
            candidates.append(y4)

        # Pick candidate minimizing the subproblem objective
        best_y = None
        best_val = float("inf")
        for y in candidates:
            val = torch.abs(torch.dot(a, y) ** 2 - b) + (beta_t / 2) * torch.norm(y - x) ** 2
            if val.item() < best_val:
                best_val = val.item()
                best_y = y

        return best_y
