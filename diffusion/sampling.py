"""Sampling Utilities"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from diffusion import extract
from diffusion.noise_schedules import karras_sigmas


def _build_sigmas(
    diffuser: "Diffuser",
    num_steps: int,
    use_karras: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (sigmas, timestep_indices) for sigma based samplers"""
    alphas_cumprod = diffuser.alphas_cumprod
    sigmas_full = ((1 - alphas_cumprod) / alphas_cumprod).sqrt()

    if use_karras:
        sigma_min = sigmas_full[0].item()
        sigma_max = sigmas_full[-1].item()
        sigmas = karras_sigmas(num_steps, sigma_min, sigma_max)
    else:
        indices = torch.linspace(len(sigmas_full) - 1, 0, num_steps).long()
        sigmas = torch.cat([sigmas_full[indices], torch.zeros(1)])

    dists = (sigmas[:-1, None] - sigmas_full[None, :]).abs()
    timesteps = dists.argmin(dim=1)

    return sigmas, timesteps

@torch.no_grad()
def step_probabilistic(
    model: nn.Module,
    diffuser: "Diffuser",
    x: torch.Tensor,
    timestep: int,
    cond_context: Optional[torch.Tensor] = None,
    uncond_context: Optional[torch.Tensor] = None,
    guidance_scale: float = 0.0,
) -> torch.Tensor:
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
def sample_probabilistic(
    model: nn.Module,
    diffuser: "Diffuser",
    shape: Tuple[int, ...],
    textencoder: Optional[nn.Module] = None,
    prompts: Optional[List[str]] = None,
    guidance_scale: float = 5.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    device = next(model.parameters()).device

    img = torch.randn(shape, device=device)

    cond_context = uncond_context = None
    if textencoder is not None and prompts is not None:
        cond_context = textencoder.encode(prompts)
        uncond_context = textencoder.encode([""] * len(prompts))

    for i in range(0, diffuser.timesteps)[::-1]:
        img = step_probabilistic(model, diffuser, img, i, cond_context, uncond_context, guidance_scale)
        if callback is not None:
            callback(diffuser.timesteps - 1 - i, img)

    return img

def cfg_combine(
    eps_uncond: torch.Tensor,
    eps_cond: torch.Tensor,
    scale: float,
    rescale: float = 0.0,
) -> torch.Tensor:
    """Classifier-free guidance with optional rescale"""
    eps_cfg = eps_uncond + scale * (eps_cond - eps_uncond)
    if rescale > 0:
        std_cfg = eps_cfg.std(dim=list(range(1, eps_cfg.ndim)), keepdim=True)
        std_cond = eps_cond.std(dim=list(range(1, eps_cond.ndim)), keepdim=True)
        eps_cfg = rescale * (eps_cfg * std_cond / (std_cfg + 1e-8)) + (1 - rescale) * eps_cfg
    return eps_cfg

@torch.no_grad()
def sample_euler_ancestral(
    model: nn.Module,
    diffuser: "Diffuser",
    shape: Tuple[int, ...],
    num_steps: int = 20,
    textencoder: Optional[nn.Module] = None,
    prompts: Optional[List[str]] = None,
    guidance_scale: float = 7.5,
    use_karras: bool = False,
    cfg_rescale: float = 0.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Euler Ancestral sampler"""
    device = next(model.parameters()).device
    sigmas, timesteps = _build_sigmas(diffuser, num_steps, use_karras=use_karras)

    cond_context = uncond_context = None
    if textencoder is not None and prompts is not None:
        cond_context = textencoder.encode(prompts).to(device)
        uncond_context = textencoder.encode([""] * len(prompts)).to(device)

    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(num_steps):
        sigma = sigmas[i].to(device)
        sigma_next = sigmas[i + 1].to(device)
        t = timesteps[i]

        model_input = x / (sigma ** 2 + 1).sqrt()
        t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)

        eps = _predict(model, model_input, t_batch, cond_context, uncond_context,
                       guidance_scale, cfg_rescale=cfg_rescale)

        denoised = x - sigma * eps

        if sigma_next > 0:
            sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2)
                        / sigma ** 2).sqrt()
            sigma_down = (sigma_next ** 2 - sigma_up ** 2).sqrt()
        else:
            sigma_up = torch.zeros_like(sigma)
            sigma_down = torch.zeros_like(sigma)

        d = (x - denoised) / sigma
        x = x + d * (sigma_down - sigma) 

        if sigma_next > 0:
            x = x + sigma_up * torch.randn_like(x)

        if callback is not None:
            callback(i, x)

        if (i + 1) % 5 == 0 or i == num_steps - 1:
            print(f"  Step {i + 1}/{num_steps}")

    return x 


def predict_noise_cfg(
    unet: nn.Module,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor,
    guidance_scale: float,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    context_in = torch.cat([uncond_context, cond_context], dim=0)

    eps_uncond, eps_cond = unet(x_in, t_in, context_in).chunk(2, dim=0)
    return cfg_combine(eps_uncond, eps_cond, guidance_scale, rescale=cfg_rescale)


def _predict(
    model: nn.Module,
    x: torch.Tensor,
    t_batch: torch.Tensor,
    cond_context: Optional[torch.Tensor],
    uncond_context: Optional[torch.Tensor],
    guidance_scale: float,
    cfg_rescale: float = 0.0,
) -> torch.Tensor:
    """run model with or without CFG."""
    if cond_context is not None and uncond_context is not None:
        return predict_noise_cfg(model, x, t_batch, cond_context, uncond_context,
                                 guidance_scale, cfg_rescale=cfg_rescale)
    return model(x, t_batch, None)


@torch.no_grad()
def sample_ddim(
    model: nn.Module,
    diffuser: "Diffuser",
    shape: Tuple[int, ...],
    num_steps: int = 50,
    eta: float = 0.0,
    textencoder: Optional[nn.Module] = None,
    prompts: Optional[List[str]] = None,
    guidance_scale: float = 7.5,
    cfg_rescale: float = 0.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """DDIM sampler"""
    device = next(model.parameters()).device
    alphas_cumprod = diffuser.alphas_cumprod.to(device)

    timesteps = torch.linspace(diffuser.timesteps - 1, 0, num_steps).long()

    cond_context = uncond_context = None
    if textencoder is not None and prompts is not None:
        cond_context = textencoder.encode(prompts).to(device)
        uncond_context = textencoder.encode([""] * len(prompts)).to(device)

    x = torch.randn(shape, device=device)

    for i, t_idx in enumerate(timesteps):
        t_prev_idx = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0)

        ac      = alphas_cumprod[t_idx]
        ac_prev = alphas_cumprod[t_prev_idx]

        alpha_t    = ac.sqrt()
        sigma_t    = (1 - ac).sqrt()
        alpha_prev = ac_prev.sqrt()

        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        eps = _predict(model, x, t_batch, cond_context, uncond_context, guidance_scale,
                       cfg_rescale=cfg_rescale)

        x0_pred = (x - sigma_t * eps) / alpha_t
        sigma_ddim = eta * ((1 - ac_prev) / (1 - ac)).sqrt() * (1 - ac / ac_prev).sqrt()

        dir_xt = (1 - ac_prev - sigma_ddim ** 2).clamp(min=0).sqrt() * eps

        noise = torch.randn_like(x) if eta > 0 else torch.zeros_like(x)
        x = alpha_prev * x0_pred + dir_xt + sigma_ddim * noise

        if callback is not None:
            callback(i, x)

        if (i + 1) % 10 == 0 or i == num_steps - 1:
            print(f"  Step {i + 1}/{num_steps}")

    return x


@torch.no_grad()
def sample_heun(
    model: nn.Module,
    diffuser: "Diffuser",
    shape: Tuple[int, ...],
    num_steps: int = 20,
    textencoder: Optional[nn.Module] = None,
    prompts: Optional[List[str]] = None,
    guidance_scale: float = 7.5,
    use_karras: bool = False,
    cfg_rescale: float = 0.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """Heun (2nd-order) sampler."""
    device = next(model.parameters()).device
    sigmas, timesteps = _build_sigmas(diffuser, num_steps, use_karras=use_karras)

    cond_context = uncond_context = None
    if textencoder is not None and prompts is not None:
        cond_context = textencoder.encode(prompts).to(device)
        uncond_context = textencoder.encode([""] * len(prompts)).to(device)

    x = torch.randn(shape, device=device) * sigmas[0]

    for i in range(num_steps):
        sigma      = sigmas[i].to(device)
        sigma_next = sigmas[i + 1].to(device)
        t_batch    = torch.full((shape[0],), timesteps[i], device=device, dtype=torch.long)

        model_input = x / (sigma ** 2 + 1).sqrt()
        eps1        = _predict(model, model_input, t_batch, cond_context, uncond_context, guidance_scale, cfg_rescale=cfg_rescale)
        denoised1   = x - sigma * eps1
        d1          = (x - denoised1) / sigma

        x_next = x + d1 * (sigma_next - sigma)

        if sigma_next > 0:
            t_next_batch = torch.full((shape[0],), timesteps[i + 1], device=device, dtype=torch.long)
            model_input2 = x_next / (sigma_next ** 2 + 1).sqrt()
            eps2         = _predict(model, model_input2, t_next_batch, cond_context, uncond_context, guidance_scale, cfg_rescale=cfg_rescale)
            denoised2    = x_next - sigma_next * eps2
            d2           = (x_next - denoised2) / sigma_next

            x = x + (d1 + d2) / 2 * (sigma_next - sigma)
        else:
            x = x_next 

        if callback is not None:
            callback(i, x)

        if (i + 1) % 5 == 0 or i == num_steps - 1:
            print(f"  Step {i + 1}/{num_steps}")

    return x


@torch.no_grad()
def sample_dpm_pp_2m(
    model: nn.Module,
    diffuser: "Diffuser",
    shape: Tuple[int, ...],
    num_steps: int = 20,
    textencoder: Optional[nn.Module] = None,
    prompts: Optional[List[str]] = None,
    guidance_scale: float = 7.5,
    use_karras: bool = False,
    cfg_rescale: float = 0.0,
    callback: Optional[Callable[[int, torch.Tensor], None]] = None,
) -> torch.Tensor:
    """DPM Solver++ (2M) sampler"""
    device = next(model.parameters()).device
    alphas_cumprod = diffuser.alphas_cumprod.to(device)

    sigmas, timesteps = _build_sigmas(diffuser, num_steps, use_karras=use_karras)

    cond_context = uncond_context = None
    if textencoder is not None and prompts is not None:
        cond_context = textencoder.encode(prompts).to(device)
        uncond_context = textencoder.encode([""] * len(prompts)).to(device)

    x = torch.randn(shape, device=device)

    D0_prev = None
    h_prev  = None

    for i in range(num_steps):
        t_idx   = timesteps[i]
        ac_s    = alphas_cumprod[t_idx]
        alpha_s = ac_s.sqrt()
        sigma_s = (1 - ac_s).sqrt()

        t_batch = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
        eps     = _predict(model, x, t_batch, cond_context, uncond_context, guidance_scale, cfg_rescale=cfg_rescale)

        D0 = (x - sigma_s * eps) / alpha_s

        if i == num_steps - 1:
            x = D0
            print(f"  Step {i + 1}/{num_steps}")
            break

        t_next_idx = timesteps[i + 1]
        ac_t    = alphas_cumprod[t_next_idx]
        alpha_t = ac_t.sqrt()
        sigma_t = (1 - ac_t).sqrt()

        lam_s = (alpha_s / sigma_s).log()
        lam_t = (alpha_t / sigma_t).log()
        h = lam_t - lam_s

        if D0_prev is None:
            x = (sigma_t / sigma_s) * x - alpha_t * torch.expm1(-h) * D0
        else:
            r  = h_prev / h
            D1 = (D0 - D0_prev) / (2 * r)
            x  = (sigma_t / sigma_s) * x - alpha_t * torch.expm1(-h) * (D0 + D1)

        D0_prev = D0
        h_prev  = h

        if callback is not None:
            callback(i, x)

        if (i + 1) % 5 == 0:
            print(f"  Step {i + 1}/{num_steps}")

    return x