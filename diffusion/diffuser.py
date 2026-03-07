"""Image Noising Utilities"""

import torch
import torch.nn.functional as F


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffuser:
    def __init__(self, timesteps, schedule):
        self.timesteps = timesteps
        self.schedule = schedule

        self.betas = self.schedule(timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def forward(self, x_start, timesteps, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sac_t = extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        somac_t = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape)

        return sac_t * x_start + somac_t * noise

    def _predict_noise(
        self,
        model,
        x_noisy: torch.Tensor,
        timesteps: torch.Tensor,
        conditionings: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if conditionings is None and condition_mask is None:
            return model(x_noisy, timesteps)

        try:
            return model(x_noisy, timesteps, conditionings=conditionings, condition_mask=condition_mask)
        except TypeError:
            # Fallback for partially updated call-sites.
            return model(x_noisy, timesteps, conditionings)

    def compute_loss(
        self,
        model,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
        conditionings: torch.Tensor | None = None,
        unconditional_conditionings: torch.Tensor | None = None,
        cond_drop_prob: float = 0.0,
        condition_mask: torch.Tensor | None = None,
        unconditional_condition_mask: torch.Tensor | None = None,
    ):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.forward(x_start, timesteps, noise)

        if not (0.0 <= cond_drop_prob <= 1.0):
            raise ValueError("cond_drop_prob must be between 0.0 and 1.0")

        cond_in = conditionings
        cond_mask_in = condition_mask

        if (
            conditionings is not None
            and unconditional_conditionings is not None
            and cond_drop_prob > 0.0
        ):
            keep_cond = torch.rand(x_start.shape[0], device=x_start.device) >= cond_drop_prob
            cond_selector = keep_cond.view(-1, *([1] * (conditionings.ndim - 1)))
            cond_in = torch.where(cond_selector, conditionings, unconditional_conditionings)

            if condition_mask is not None and unconditional_condition_mask is not None:
                mask_selector = keep_cond.view(-1, *([1] * (condition_mask.ndim - 1)))
                cond_mask_in = torch.where(mask_selector, condition_mask, unconditional_condition_mask)

        preds = self._predict_noise(
            model,
            x_noisy,
            timesteps,
            conditionings=cond_in,
            condition_mask=cond_mask_in,
        )

        loss = F.l1_loss(noise, preds) # Potentially different loss function?

        return loss


if __name__ == "__main__":
    from diffusion import linear_beta
    from matplotlib import pyplot as plt

    n = Diffuser(100, linear_beta)
    im = torch.zeros((1, 64, 64))
    step = 20
    noised = n.forward(im, torch.tensor([step]))

    plt.imshow(noised.squeeze().numpy(), vmin=-1, vmax=1)
    plt.show()
