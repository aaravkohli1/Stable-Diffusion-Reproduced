"""Sampling Utilities"""

import torch
from diffusion import extract

@torch.no_grad()
def step_probabilistic(
    model,
    diffuser,
    x,
    timestep,
    cond_context=None,
    uncond_context=None,
    guidance_scale=0.0,
    guidance_strategy="anchored",
    ref_context=None,
):
    device = next(model.parameters()).device

    t = torch.full((x.shape[0],), timestep, device=device, dtype=torch.long)

    betas_t = extract(diffuser.betas, t, x.shape)
    somac_t = extract(diffuser.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sra_t = extract(diffuser.sqrt_recip_alphas, t, x.shape)

    if ref_context is None:
        ref_context = uncond_context

    if cond_context is not None and ref_context is not None:
        eps = predict_noise_cfg(
            unet=model,
            x_t=x,
            t=t,
            cond_context=cond_context,
            ref_context=ref_context,
            guidance_scale=guidance_scale,
            guidance_strategy=guidance_strategy,
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
    model,
    diffuser,
    shape,
    text_encoder=None,
    prompts=None,
    negative_prompts=None,
    guidance_scale=5.0,
    guidance_strategy="anchored",
    textencoder=None,
):
    device = next(model.parameters()).device

    img = torch.randn(shape, device=device)
    cond_context = None
    ref_context = None

    if text_encoder is None:
        text_encoder = textencoder

    if text_encoder is not None and prompts is not None:
        prompt_list = _normalize_prompt_list(prompts, shape[0], "prompts")
        cond_context = text_encoder.encode(prompt_list)

        if negative_prompts is None:
            reference_prompt_list = [""] * len(prompt_list)
        else:
            reference_prompt_list = _normalize_prompt_list(negative_prompts, shape[0], "negative_prompts")
        ref_context = text_encoder.encode(reference_prompt_list)

    for i in range(0, diffuser.timesteps)[::-1]:
        img = step_probabilistic(
            model,
            diffuser,
            img,
            i,
            cond_context=cond_context,
            uncond_context=ref_context,
            guidance_scale=guidance_scale,
            guidance_strategy=guidance_strategy,
        )

    return img

def _normalize_prompt_list(prompt_input, batch_size: int, name: str) -> list[str]:
    if isinstance(prompt_input, str):
        prompts = [prompt_input]
    else:
        prompts = list(prompt_input)

    if len(prompts) == 1 and batch_size > 1:
        prompts = prompts * batch_size

    if len(prompts) != batch_size:
        raise ValueError(
            f"{name} length ({len(prompts)}) must match batch size ({batch_size}), "
            "or provide a single prompt to broadcast."
        )
    return prompts

def combine_guidance(
    eps_cond: torch.Tensor,
    eps_ref: torch.Tensor,
    scale: float,
    strategy: str = "anchored",
) -> torch.Tensor:
    if strategy == "anchored":
        return eps_ref + scale * (eps_cond - eps_ref)
    if strategy == "difference":
        return eps_cond - scale * eps_ref
    raise ValueError(f"Unknown guidance strategy: {strategy}")

def cfg_combine(eps_uncond: torch.Tensor, eps_cond: torch.Tensor, scale: float) -> torch.Tensor:
    return combine_guidance(
        eps_cond=eps_cond,
        eps_ref=eps_uncond,
        scale=scale,
        strategy="anchored",
    )

def predict_noise_cfg(
    unet,
    x_t: torch.Tensor,
    t: torch.Tensor,
    cond_context: torch.Tensor,
    uncond_context: torch.Tensor | None = None,
    guidance_scale: float = 0.0,
    guidance_strategy: str = "anchored",
    ref_context: torch.Tensor | None = None,
) -> torch.Tensor:
    if ref_context is None:
        ref_context = uncond_context
    if ref_context is None:
        raise ValueError("ref_context or uncond_context must be provided for CFG.")

    x_in = torch.cat([x_t, x_t], dim=0)
    t_in = torch.cat([t, t], dim=0)
    context_in = torch.cat([ref_context, cond_context], dim=0)

    eps_ref, eps_cond = unet(x_in, t_in, context_in).chunk(2, dim=0)
    return combine_guidance(
        eps_cond=eps_cond,
        eps_ref=eps_ref,
        scale=guidance_scale,
        strategy=guidance_strategy,
    )
