import torch
from diffusion.sampling import cfg_combine

def test_cfg_scale_zero_is_unconditional():
    eps_uncond = torch.randn(2, 4, 8, 8)
    eps_cond = torch.randn(2, 4, 8, 8)

    out = cfg_combine(eps_uncond, eps_cond, scale=0.0)
    assert torch.allclose(out, eps_uncond)

def test_cfg_scale_one_is_conditional():
    eps_uncond = torch.randn(2, 4, 8, 8)
    eps_cond = torch.randn(2, 4, 8, 8)

    out = cfg_combine(eps_uncond, eps_cond, scale=1.0)
    assert torch.allclose(out, eps_cond)