"""Noise Scheduling Utilities"""

import torch

def cosine_beta(timesteps):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos((x / timesteps) * torch.pi * 0.5) ** 2
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) * 0.02

def linear_beta(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    t = 100

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(torch.linspace(0, t, t), linear_beta(t), c='r', label='Linear Beta')
    ax.scatter(torch.linspace(0, t, t), quadratic_beta(t), c='b', label='Quadratic Beta')
    ax.scatter(torch.linspace(0, t, t), cosine_beta(t), c='g', label='Cosine Beta')

    plt.legend()
    plt.show()
