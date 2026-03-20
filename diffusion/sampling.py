"""Sampling Utilities"""

import torch
from diffusion import extract

@torch.no_grad()
def step_probabilistic(model, diffuser, x, timestep, cond_context=None, uncond_context=None, guidance_scale=0.0):
    device = next(model.parameters()).device

    t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)

    betas_t = extract(diffuser.betas, t, x.shape)
    somac_t = extract(diffuser.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sra_t = extract(diffuser.sqrt_recip_alphas, t, x.shape)

    if cond_context is not None and uncond_context is not None:
        eps = predict_noise_cfg(
            unet=model,
            x_t=x,
            t=t,
            cond_context=cond_context,
            uncond_context=uncond_context,
            guidance_scale=guidance_scale,
        )
    else:
        eps = model(x, t, None)

    model_mean = sra_t * (x - betas_t * eps / somac_t)

    if timestep == 0:
        return model_mean
    else:
        # Add back noise as per DDPM paper
        posterior_variance_t = extract(diffuser.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_probabilistic(model, diffuser, shape, textencoder=None, prompts=None, guidance_scale=5.0):
    device = next(model.parameters()).device

    img = torch.randn(shape, device=device)

    if text_encoder is not None and prompts is not None:
        cond_context = text_encoder.encode(prompts)
        uncond_context = text_encoder.encode([""] * len(prompts))

    for i in range(0, diffuser.timesteps)[::-1]:
        img = step_probabilistic(model, diffuser, img, i, cond_context, uncond_context, guidance_scale)

    return img

def cfg_combine(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    return eps_uncond + scale * (eps_cond - eps_uncond)

def predict_noise_cfg(
    unet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    context_in = torch.cat([uncond_context, cond_context], dim=0)

    eps_uncond, eps_cond = unet(x_in, t_in, context_in).chunk(2, dim=0)
    return cfg_combine(eps_uncond, eps_cond, guidance_scale)