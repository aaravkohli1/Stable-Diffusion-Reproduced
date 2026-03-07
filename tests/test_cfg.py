import torch
import torch.nn as nn

from diffusion import Diffuser, linear_beta
from diffusion.sampling import step_probabilistic


class _ConditionedToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))

    def forward(self, x, t, conditionings=None, condition_mask=None):
        if conditionings is None:
            return torch.zeros_like(x)
        values = conditionings.mean(dim=(1, 2), keepdim=True).view(-1, 1, 1, 1)
        return torch.ones_like(x) * values


class _CaptureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.last_conditionings = None

    def forward(self, x, t, conditionings=None, condition_mask=None):
        self.last_conditionings = conditionings
        return torch.zeros_like(x)


def test_cfg_guidance_equation_timestep_zero():
    model = _ConditionedToyModel()
    diffuser = Diffuser(timesteps=10, schedule=linear_beta)

    x = torch.randn(2, 4, 8, 8)
    cond = torch.ones(2, 77, 512) * 3.0
    uncond = torch.ones(2, 77, 512) * 1.0
    guidance_scale = 2.5
    t = 0

    out = step_probabilistic(
        model=model,
        diffuser=diffuser,
        x=x,
        timestep=t,
        conditionings=cond,
        unconditional_conditionings=uncond,
        guidance_scale=guidance_scale,
    )

    # At timestep=0, there is no added posterior noise in step_probabilistic.
    tt = torch.zeros((x.shape[0],), dtype=torch.long)
    betas_t = diffuser.betas.gather(0, tt).view(-1, 1, 1, 1)
    somac_t = diffuser.sqrt_one_minus_alphas_cumprod.gather(0, tt).view(-1, 1, 1, 1)
    sra_t = diffuser.sqrt_recip_alphas.gather(0, tt).view(-1, 1, 1, 1)

    eps_cond = torch.ones_like(x) * 3.0
    eps_uncond = torch.ones_like(x) * 1.0
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    expected = sra_t * (x - betas_t * eps / somac_t)

    assert torch.allclose(out, expected, atol=1e-5)


def test_compute_loss_cond_dropout_uses_unconditional_embeddings():
    model = _CaptureModel()
    diffuser = Diffuser(timesteps=10, schedule=linear_beta)

    x_start = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 10, (2,), dtype=torch.long)
    cond = torch.randn(2, 77, 512)
    uncond = torch.randn(2, 77, 512)

    _ = diffuser.compute_loss(
        model=model,
        x_start=x_start,
        timesteps=t,
        noise=torch.zeros_like(x_start),
        conditionings=cond,
        unconditional_conditionings=uncond,
        cond_drop_prob=1.0,
    )

    assert model.last_conditionings is not None
    assert torch.allclose(model.last_conditionings, uncond)
