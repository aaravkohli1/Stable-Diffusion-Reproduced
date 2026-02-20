"""Sampling Utilities"""

import torch
from diffusion import extract

@torch.no_grad()
def step_probabilistic(model, diffuser, x, timestep):
    device = next(model.parameters()).device

    t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)

    betas_t = extract(diffuser.betas, t, x.shape)
    somac_t = extract(diffuser.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sra_t = extract(diffuser.sqrt_recip_alphas, t, x.shape)

    model_mean = sra_t * (x - betas_t * model(x, t) / somac_t)

    if timestep == 0:
        return model_mean
    else:
        # Add back noise as per DDPM paper
        posterior_variance_t = extract(diffuser.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_probabilistic(model, diffuser, shape):
    device = next(model.parameters()).device

    img = torch.randn(shape, device=device)

    for i in range(0, diffuser.timesteps)[::-1]:
        img = step_probabilistic(model, diffuser, img, i)

    return img
