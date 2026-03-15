import torch

from diffusion.diffuser import Diffuser
from diffusion.noise_schedules import linear_beta
from diffusion.sampling import sample_probabilistic


class DummyTextEncoder:
    def __init__(self, table):
        self.table = table

    def encode(self, prompts):
        return torch.stack([self.table[p] for p in prompts], dim=0)


class DummyUNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t, context=None):
        if context is None:
            return torch.zeros_like(x)

        score = context.mean(dim=(1, 2), keepdim=True).view(-1, 1, 1, 1).to(x.dtype)
        return torch.zeros_like(x) + score


def build_dummy_components():
    prompt_table = {
        "a cat": torch.full((2, 3), 2.0),
        "": torch.zeros((2, 3)),
        "blurry": torch.full((2, 3), -2.0),
    }
    text_encoder = DummyTextEncoder(prompt_table)
    model = DummyUNet()
    diffuser = Diffuser(timesteps=1, schedule=linear_beta)
    return model, diffuser, text_encoder


def test_negative_prompt_changes_anchored_output():
    model, diffuser, text_encoder = build_dummy_components()
    shape = (1, 4, 8, 8)

    torch.manual_seed(7)
    out_standard = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=shape,
        text_encoder=text_encoder,
        prompts=["a cat"],
        guidance_scale=2.0,
        guidance_strategy="anchored",
    )

    torch.manual_seed(7)
    out_negative = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=shape,
        text_encoder=text_encoder,
        prompts=["a cat"],
        negative_prompts=["blurry"],
        guidance_scale=2.0,
        guidance_strategy="anchored",
    )

    assert not torch.allclose(out_standard, out_negative)
    assert out_negative.mean() < out_standard.mean()


def test_difference_strategy_runs_and_differs_from_anchored():
    model, diffuser, text_encoder = build_dummy_components()
    shape = (1, 4, 8, 8)

    torch.manual_seed(11)
    out_anchored = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=shape,
        text_encoder=text_encoder,
        prompts=["a cat"],
        negative_prompts=["blurry"],
        guidance_scale=1.0,
        guidance_strategy="anchored",
    )

    torch.manual_seed(11)
    out_difference = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=shape,
        text_encoder=text_encoder,
        prompts=["a cat"],
        negative_prompts=["blurry"],
        guidance_scale=1.0,
        guidance_strategy="difference",
    )

    assert not torch.allclose(out_anchored, out_difference)


def test_no_text_path_is_unconditional_and_stable():
    model, diffuser, _ = build_dummy_components()

    out = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=(2, 4, 8, 8),
        guidance_scale=5.0,
    )

    assert out.shape == (2, 4, 8, 8)


def test_legacy_textencoder_keyword_is_supported():
    model, diffuser, text_encoder = build_dummy_components()

    out = sample_probabilistic(
        model=model,
        diffuser=diffuser,
        shape=(1, 4, 8, 8),
        textencoder=text_encoder,
        prompts=["a cat"],
    )

    assert out.shape == (1, 4, 8, 8)
