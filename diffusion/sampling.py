"""Sampling Utilities"""

import torch
from .diffuser import extract


def _predict_noise(
    model,
    x: torch.Tensor,
    timesteps: torch.Tensor,
    conditionings: torch.Tensor | None = None,
    condition_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if conditionings is None and condition_mask is None:
        return model(x, timesteps)

    try:
        return model(x, timesteps, conditionings=conditionings, condition_mask=condition_mask)
    except TypeError:
        # Fallback for partially updated call-sites.
        return model(x, timesteps, conditionings)


@torch.no_grad()
def step_probabilistic(
    model,
    diffuser,
    x,
    timestep,
    conditionings: torch.Tensor | None = None,
    unconditional_conditionings: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    condition_mask: torch.Tensor | None = None,
    unconditional_condition_mask: torch.Tensor | None = None,
):
    device = next(model.parameters()).device

    t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)

    betas_t = extract(diffuser.betas, t, x.shape)
    somac_t = extract(diffuser.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sra_t = extract(diffuser.sqrt_recip_alphas, t, x.shape)

    if conditionings is None:
        noise_pred = _predict_noise(model, x, t)
    elif unconditional_conditionings is None or guidance_scale == 0.0:
        noise_pred = _predict_noise(model, x, t, conditionings=conditionings, condition_mask=condition_mask)
    else:
        noise_uncond = _predict_noise(
            model,
            x,
            t,
            conditionings=unconditional_conditionings,
            condition_mask=unconditional_condition_mask,
        )
        noise_cond = _predict_noise(
            model,
            x,
            t,
            conditionings=conditionings,
            condition_mask=condition_mask,
        )
        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

    model_mean = sra_t * (x - betas_t * noise_pred / somac_t)

    if timestep == 0:
        return model_mean
    else:
        # Add back noise as per DDPM paper
        posterior_variance_t = extract(diffuser.posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

@torch.no_grad()
def sample_probabilistic(
    model,
    diffuser,
    shape,
    conditionings: torch.Tensor | None = None,
    unconditional_conditionings: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    condition_mask: torch.Tensor | None = None,
    unconditional_condition_mask: torch.Tensor | None = None,
):
    device = next(model.parameters()).device

    img = torch.randn(shape, device=device)

    for i in range(0, diffuser.timesteps)[::-1]:
        img = step_probabilistic(
            model,
            diffuser,
            img,
            i,
            conditionings=conditionings,
            unconditional_conditionings=unconditional_conditionings,
            guidance_scale=guidance_scale,
            condition_mask=condition_mask,
            unconditional_condition_mask=unconditional_condition_mask,
        )

    return img
