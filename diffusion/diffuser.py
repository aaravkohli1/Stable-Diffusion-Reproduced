"""Image Noising Utilities"""

import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffuser:
    def __init__(self, timesteps, schedule):
        self.timesteps = timesteps
        self.schedule = schedule

        self.betas = self.schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sac_t = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        somac_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)

        return sac_t * x_start + somac_t * noise

    def compute_loss(self, model, x_start, timesteps, noise=None, context=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.forward(x_start, timesteps, noise)
        if context is None:
            preds = model(x_noisy, timesteps)
        else:
            preds = model(x_noisy, timesteps, context)

        loss = F.l1_loss(noise, preds) # Potentially different loss function?

        return loss


if __name__ == "__main__":
    from diffusion import linear_beta
    from matplotlib import pyplot as plt

    n = Diffuser(100, linear_beta)
    im = torch.zeros((1, 64, 64))
    step = 20
    noised = n.forward(im, torch.tensor([step]))

    plt.imshow(noised.squeeze().numpy(), vmin=-1, vmax=1)
    plt.show()
