"""
Blind Deconvolution problem (Section 5.2 of Davis & Drusvyatskiy 2019).

Objective: min_{x,y} (1/m) Σ |<u_i, x><v_i, y> - b_i|

Variable z = [x; y] ∈ R^{d1+d2}, where x = z[:d1], y = z[d1:].
Data layout: each row = [u_i (d1 cols), v_i (d2 cols), b_i (1 col)].
"""

import torch
import numpy as np
from src.WeaklyConvexProblem import WeaklyConvexProblem


def f_blind_deconv(z, batch, d1):
    """g(x,y) = |<u,x><v,y> - b| averaged over the batch."""
    x = z[:d1]
    y = z[d1:]
    u = batch[:, :d1]
    v = batch[:, d1:-1]
    b = batch[:, -1]
    return torch.abs((u @ x) * (v @ y) - b)


class BlindDeconvSubgradient(WeaklyConvexProblem):
    """
    Stochastic subgradient method for blind deconvolution (Section 5.2).

    ∂g(x,y) = (<v,y>u, <u,x>v) · sign(<u,x><v,y> - b).
    Update: z ← z - (1/β) * subgradient.
    """

    def __init__(self, d1, rho=2.0):
        self.d1 = d1
        super().__init__(
            f_objective=lambda z, batch: f_blind_deconv(z, batch, d1),
            r_objective=lambda z: 0,
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, z_t, batch, beta_t):
        x = z_t[: self.d1]
        y = z_t[self.d1 :]
        lam = 1.0 / beta_t

        grad_sum = torch.zeros_like(z_t)
        for i in range(batch.shape[0]):
            u = batch[i, : self.d1]
            v = batch[i, self.d1 : -1]
            b = batch[i, -1]

            ux = torch.dot(u, x)
            vy = torch.dot(v, y)
            residual = ux * vy - b
            sign = torch.sign(residual) if residual.abs() > 1e-12 else torch.tensor(0.0)

            # ∂g = sign * (<v,y>u, <u,x>v)
            grad_x = sign * vy * u
            grad_y = sign * ux * v
            grad_sum[: self.d1] += grad_x
            grad_sum[self.d1 :] += grad_y

        grad_avg = grad_sum / batch.shape[0]
        return z_t - lam * grad_avg


class BlindDeconvProxLinear(WeaklyConvexProblem):
    """
    Stochastic prox-linear method for blind deconvolution (Section 5.2).

    Uses eq. (5.2) with ζ = λ(<v,y>u, <u,x>v), γ = λ(<u,x><v,y> - b).
    """

    def __init__(self, d1, rho=2.0):
        self.d1 = d1
        super().__init__(
            f_objective=lambda z, batch: f_blind_deconv(z, batch, d1),
            r_objective=lambda z: 0,
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, z_t, batch, beta_t):
        x = z_t[: self.d1]
        y = z_t[self.d1 :]

        delta_sum = torch.zeros_like(z_t)
        count = 0

        for i in range(batch.shape[0]):
            u = batch[i, : self.d1]
            v = batch[i, self.d1 : -1]
            b = batch[i, -1]

            ux = torch.dot(u, x)
            vy = torch.dot(v, y)
            gamma = ux * vy - b
            # ζ = λ(<v,y>u, <u,x>v)  but λ cancels in the formula
            # Using eq (5.2): Δ* = proj_{[-1,1]}(-γβ/||ζ||²) · ζ / β
            zeta = torch.cat([vy * u, ux * v])
            norm_zeta_sq = torch.dot(zeta, zeta)

            if norm_zeta_sq < 1e-9:
                continue

            scale = torch.clamp(-gamma * beta_t / norm_zeta_sq, min=-1.0, max=1.0)
            delta_sum += (scale / beta_t) * zeta
            count += 1

        if count == 0:
            return z_t

        return z_t + delta_sum / count


class BlindDeconvProximalPoint(WeaklyConvexProblem):
    """
    Stochastic proximal point method for blind deconvolution (Section 5.2).

    Subproblem: argmin_{x,y} |<u,x><v,y>-b| + (1/2λ)||x-x₀||² + (1/2λ)||y-y₀||²

    Case 1 (<u,x><v,y> ≠ b): eq. (5.6), two candidates from ± sign.
    Case 2 (<u,x><v,y> = b): solve quartic for η, then recover (x,y) via eq. (5.8).
    """

    def __init__(self, d1, rho=2.0):
        self.d1 = d1
        super().__init__(
            f_objective=lambda z, batch: f_blind_deconv(z, batch, d1),
            r_objective=lambda z: 0,
            model_gen=None,
            rho=rho,
        )

    def solve_exact(self, z_t, batch, beta_t):
        lam = 1.0 / beta_t
        z_new = z_t.clone()
        for i in range(batch.shape[0]):
            u = batch[i, : self.d1]
            v = batch[i, self.d1 : -1]
            b = batch[i, -1]
            z_new = self._solve_single(z_new, u, v, b, lam, beta_t)
        return z_new

    def _solve_single(self, z, u, v, b, lam, beta_t):
        x0 = z[: self.d1]
        y0 = z[self.d1 :]

        ux0 = torch.dot(u, x0)
        vy0 = torch.dot(v, y0)
        norm_u_sq = torch.dot(u, u)
        norm_v_sq = torch.dot(v, v)

        candidates = []

        # Case 1: <u,x><v,y> ≠ b → eq (5.6), ± sign
        denom = 1.0 - lam**2 * norm_u_sq * norm_v_sq
        if abs(denom.item()) > 1e-9:
            for sign in [1.0, -1.0]:
                num_x = sign * vy0 - lam * norm_v_sq * ux0
                num_y = sign * ux0 - lam * norm_u_sq * vy0
                x_cand = x0 - lam * (num_x / denom) * u
                y_cand = y0 - lam * (num_y / denom) * v
                candidates.append(torch.cat([x_cand, y_cand]))

        # Case 2: <u,x><v,y> = b → solve quartic for η
        # 0 = η⁴||v||² - η³||v||²<u,x₀> + bη||u||²<v,y₀> - b²||u||²
        coeffs = [
            norm_v_sq.item(),
            -norm_v_sq.item() * ux0.item(),
            0.0,
            b.item() * norm_u_sq.item() * vy0.item(),
            -b.item() ** 2 * norm_u_sq.item(),
        ]

        roots = np.roots(coeffs)
        for root in roots:
            if np.iscomplex(root) and abs(root.imag) > 1e-8:
                continue
            eta = float(root.real)
            if abs(b.item()) < 1e-12:
                continue
            gamma_val = (eta * ux0.item() - eta**2) / (b.item() * norm_u_sq.item())
            delta = eta  # δ = <v,y> = b/η when b ≠ 0... Actually δ = b/η
            if abs(eta) < 1e-12:
                continue
            delta = b.item() / eta
            x_cand = x0 - gamma_val * delta * u
            y_cand = y0 - gamma_val * eta * v
            candidates.append(torch.cat([x_cand.detach(), y_cand.detach()]))

        # Pick the best candidate
        best_z = z
        best_val = float("inf")
        for z_cand in candidates:
            x_c = z_cand[: self.d1]
            y_c = z_cand[self.d1 :]
            obj = torch.abs(torch.dot(u, x_c) * torch.dot(v, y_c) - b) + (
                beta_t / 2
            ) * (torch.norm(x_c - x0) ** 2 + torch.norm(y_c - y0) ** 2)
            if obj.item() < best_val:
                best_val = obj.item()
                best_z = z_cand

        return best_z
