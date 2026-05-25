"""Precompute VAE latents and CLIP text embeddings.

Caching these avoids re-running the VAE encoder and CLIP text encoder on every UNet
training step, which is the dominant cost when training small UNets on a single GPU.

Each cached sample is saved as a ``.pt`` containing:
    {
        "latents": Tensor[4, H/8, W/8] in latent-space scale,
        "conds":   Tensor[77, d_cond] text embedding,
    }

Example:
    python -m runs.compute_latents --out runs/latent_cache --num-samples 200 --testing
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data import DiffusionDataset
from models.vae import VAE
from models.clip.text_encoder import CLIPTextEncoder


def get_device(requested: str | None = None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/latent_cache")
    p.add_argument("--testing", action="store_true", help="Use CIFAR-10 wrapper instead of LAION shards")
    p.add_argument("--num-samples", type=int, default=200)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--vae-ckpt", default=None, help="Path to a trained VAE checkpoint")
    p.add_argument("--clip", default="openai/clip-vit-base-patch32")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    print(f"Using device: {device}")

    vae = VAE().to(device).eval()
    if args.vae_ckpt:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device, weights_only=True))
        print(f"Loaded VAE: {args.vae_ckpt}")

    text_encoder = CLIPTextEncoder.from_pretrained_hf(args.clip).freeze().to(device)

    dataset = DiffusionDataset(testing=args.testing, num_test_samples=args.num_samples, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    written = 0
    for i, batch in enumerate(loader):
        if written >= args.num_samples:
            break
        image = batch["image"].to(device)
        caption = batch["caption"][0] if isinstance(batch["caption"], list) else batch["caption"]
        _, encoded, _, _ = vae(image)
        cond = text_encoder.encode([caption]).cpu()

        sample = {"latents": encoded.squeeze(0).cpu(), "conds": cond.squeeze(0)}
        path = os.path.join(args.out, f"sample_{i:06d}.pt")
        torch.save(sample, path)
        written += 1
        if written % 20 == 0:
            print(f"  cached {written}/{args.num_samples}")

    print(f"done — wrote {written} samples to {args.out}")


if __name__ == "__main__":
    main()
