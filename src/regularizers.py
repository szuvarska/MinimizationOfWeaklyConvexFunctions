import torch


def l1_regularizer(x):
    return 0.5 * torch.norm(x, p=1)  # Encourages sparsity


def l1_prox(x, thr):
    """
    Proximal operator of thr * ||.||_1 (soft-thresholding).

    prox_{thr ||.||_1}(x) = sign(x) * max(|x| - thr, 0).

    Used by the proximal stochastic subgradient method when r(x) = mu*||x||_1:
        x_{t+1} = prox_{alpha*mu*||.||_1}(x_t - alpha * v).
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - thr, min=0.0)


def elastic_net_regularizer(x, l1_ratio=0.5, mu=0.01):
    """
    Combines L1 and L2 regularization.
    Matches the 'regularized population risk' framework.
    """
    l1_term = l1_ratio * torch.norm(x, p=1)
    l2_term = (1 - l1_ratio) * 0.5 * torch.norm(x, p=2) ** 2
    # Ensure the result is a tensor attached to the same device as x
    return mu * (l1_term + l2_term)
