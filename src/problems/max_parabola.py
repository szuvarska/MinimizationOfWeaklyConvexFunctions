import torch


def max_parabola_phi(x, batch):
    """The 'True' Objective: max(x^2, (x-4)^2)"""
    f1 = (x - batch[:, 0]) ** 2
    f2 = (x - batch[:, 1]) ** 2
    return torch.max(f1, f2)


def max_parabola_model_gen(x_t, batch):
    """
    Returns the Stochastic Model for the Pointwise Max.
    Based on Example 2.6 and Section 4.2[cite: 984].
    """
    # Detach constants for the inner loop
    x_t_val = x_t.detach()
    f1_vals = (x_t_val - batch[:, 0]) ** 2
    f2_vals = (x_t_val - batch[:, 1]) ** 2

    g1_vals = 2 * (x_t_val - batch[:, 0])
    g2_vals = 2 * (x_t_val - batch[:, 1])

    def model(y):
        # Linearize components for each sample in batch, then average
        l1 = f1_vals + g1_vals * (y - x_t_val)
        l2 = f2_vals + g2_vals * (y - x_t_val)
        return torch.mean(torch.max(l1, l2))

    return model
