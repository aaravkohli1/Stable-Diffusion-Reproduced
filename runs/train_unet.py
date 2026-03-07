"""Minimal UNet training script with CLIP conditioning and CFG dropout."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import DiffusionDataset
from diffusion import Diffuser, linear_beta
from models.clip import CLIPTextEncoder
from models.unet import UNet
from models.vae import VAE


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditioned UNet.")
    parser.add_argument("--dataset-path", type=str, default="", help="Dataset root path containing split folders.")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--testing", action="store_true", help="Use CIFAR-10 wrapper test data.")
    parser.add_argument("--num-test-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--cond-drop-prob", type=float, default=0.1)
    parser.add_argument("--clip-model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--vae-ckpt", type=str, default="")
    parser.add_argument("--save-every", type=int, default=0, help="0 disables periodic checkpoints.")
    parser.add_argument("--output-dir", type=str, default="runs/checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto|cpu|cuda|mps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    dataset = DiffusionDataset(
        path=args.dataset_path,
        split=args.split,
        image_size=args.image_size,
        testing=args.testing,
        num_test_samples=args.num_test_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_iter = iter(loader)

    vae = VAE().to(device).eval()
    if args.vae_ckpt:
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device))
        print(f"Loaded VAE checkpoint: {args.vae_ckpt}")
    for p in vae.parameters():
        p.requires_grad = False

    text_encoder = CLIPTextEncoder.from_pretrained_hf(model_name=args.clip_model).to(device).eval()
    for p in text_encoder.parameters():
        p.requires_grad = False

    unet = UNet(in_channels=4, out_channels=4, d_cond=512).to(device)
    diffuser = Diffuser(args.timesteps, linear_beta)
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.max_steps + 1):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        images = batch["image"].to(device)
        captions = list(batch["caption"])

        with torch.no_grad():
            latents, _, _ = vae.encoder(images)
            cond, uncond = text_encoder.encode_conditioning(captions)

        timesteps = torch.randint(0, diffuser.timesteps, (latents.shape[0],), device=device)
        loss = diffuser.compute_loss(
            model=unet,
            x_start=latents,
            timesteps=timesteps,
            conditionings=cond,
            unconditional_conditionings=uncond,
            cond_drop_prob=args.cond_drop_prob,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0 or step == 1:
            print(f"step={step:04d} loss={loss.item():.6f}")

        if args.save_every > 0 and step % args.save_every == 0:
            ckpt_path = out_dir / f"unet_step_{step}.pt"
            torch.save(unet.state_dict(), ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")

    final_path = out_dir / "unet_final.pt"
    torch.save(unet.state_dict(), final_path)
    print(f"saved final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
