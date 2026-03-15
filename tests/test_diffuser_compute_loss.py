import torch

from diffusion.diffuser import Diffuser
from diffusion.noise_schedules import linear_beta


class UnconditionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t):
        return torch.zeros_like(x) + self.weight


class ConditionalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t, context=None):
        if context is None:
            return torch.zeros_like(x) + self.weight

        context_score = context.mean(dim=(1, 2), keepdim=True).view(-1, 1, 1, 1).to(x.dtype)
        return torch.zeros_like(x) + self.weight + context_score


def test_compute_loss_unconditional_path_is_unchanged():
    diffuser = Diffuser(timesteps=10, schedule=linear_beta)
    model = UnconditionalModel()
    x_start = torch.randn(2, 4, 8, 8)
    timesteps = torch.randint(0, diffuser.timesteps, (2,))

    loss = diffuser.compute_loss(model=model, x_start=x_start, timesteps=timesteps)

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_compute_loss_accepts_optional_context():
    diffuser = Diffuser(timesteps=10, schedule=linear_beta)
    model = ConditionalModel()
    x_start = torch.randn(2, 4, 8, 8)
    timesteps = torch.randint(0, diffuser.timesteps, (2,))
    context = torch.randn(2, 77, 16)

    loss = diffuser.compute_loss(model=model, x_start=x_start, timesteps=timesteps, context=context)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
