import pytest
import torch

from diffusion.sampling import combine_guidance, cfg_combine


def test_combine_guidance_anchored_scale_zero_returns_reference():
    eps_cond = torch.randn(2, 4, 8, 8)
    eps_ref = torch.randn(2, 4, 8, 8)

    out = combine_guidance(eps_cond, eps_ref, scale=0.0, strategy="anchored")
    assert torch.allclose(out, eps_ref)


def test_combine_guidance_anchored_scale_one_returns_conditional():
    eps_cond = torch.randn(2, 4, 8, 8)
    eps_ref = torch.randn(2, 4, 8, 8)

    out = combine_guidance(eps_cond, eps_ref, scale=1.0, strategy="anchored")
    assert torch.allclose(out, eps_cond)


def test_combine_guidance_difference_scale_one_subtracts_reference():
    eps_cond = torch.randn(2, 4, 8, 8)
    eps_ref = torch.randn(2, 4, 8, 8)

    out = combine_guidance(eps_cond, eps_ref, scale=1.0, strategy="difference")
    assert torch.allclose(out, eps_cond - eps_ref)


def test_combine_guidance_rejects_unknown_strategy():
    eps_cond = torch.randn(1, 4, 4, 4)
    eps_ref = torch.randn(1, 4, 4, 4)

    with pytest.raises(ValueError):
        combine_guidance(eps_cond, eps_ref, scale=1.0, strategy="bad_strategy")


def test_cfg_combine_compatibility_wrapper():
    eps_uncond = torch.randn(2, 4, 8, 8)
    eps_cond = torch.randn(2, 4, 8, 8)
    scale = 2.5

    expected = combine_guidance(eps_cond, eps_uncond, scale=scale, strategy="anchored")
    out = cfg_combine(eps_uncond, eps_cond, scale=scale)
    assert torch.allclose(out, expected)
