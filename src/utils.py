import torch


def sign_invariant_dist(x, x_ref):
    """
    Distance between x and x_ref up to a global sign flip.

    Phase retrieval and blind deconvolution recover the target signal only
    up to a global sign (x and -x give the same |<a,x>^2 - b|), so the
    natural recovery metric is min(||x - x_ref||, ||x + x_ref||).
    """
    with torch.no_grad():
        d_pos = torch.norm(x - x_ref)
        d_neg = torch.norm(x + x_ref)
        return torch.min(d_pos, d_neg).item()


def spectral_init(data, d):
    """
    Spectral initialization for phase retrieval: leading eigenvector of
    (1/m) sum_i b_i a_i a_i^T, scaled by sqrt(mean b). Standard warm start
    that lands in the basin of attraction (sign is arbitrary).
    """
    a = data[:, :-1]
    b = data[:, -1]
    m = a.shape[0]
    Y = (a.T @ (b.unsqueeze(1) * a)) / m
    _, evecs = torch.linalg.eigh(Y)
    v = evecs[:, -1]
    scale = torch.sqrt(torch.clamp(torch.mean(b), min=1e-12))
    return scale * v
