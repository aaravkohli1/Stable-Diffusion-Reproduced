"""Precompute VAE latents for a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1]))

from data.dataset import DiffusionDataset
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
    parser = argparse.ArgumentParser(description="Precompute VAE latents.")
    parser.add_argument("--dataset-path", type=str, default="", help="Dataset root path containing split folders.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split.")
    parser.add_argument("--output", type=str, default="runs/latents.pt", help="Output .pt file path.")
    parser.add_argument("--vae-ckpt", type=str, default="", help="Optional path to VAE checkpoint.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--testing", action="store_true", help="Use CIFAR-10 wrapper test data.")
    parser.add_argument("--num-test-samples", type=int, default=32)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means all batches.")
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    vae = VAE().to(device).eval()
    if args.vae_ckpt:
        state = torch.load(args.vae_ckpt, map_location=device)
        vae.load_state_dict(state)
        print(f"Loaded VAE checkpoint: {args.vae_ckpt}")

    latent_batches = []
    captions: list[str] = []
    image_paths: list[str] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images = batch["image"].to(device)
            latents, _, _ = vae.encoder(images)
            latent_batches.append(latents.cpu())
            captions.extend(batch["caption"])
            image_paths.extend(batch["image_path"])

            if (batch_idx + 1) % 10 == 0:
                print(f"Processed batches: {batch_idx + 1}")

            if args.max_batches > 0 and (batch_idx + 1) >= args.max_batches:
                break

    latents = torch.cat(latent_batches, dim=0) if latent_batches else torch.empty(0)
    payload = {
        "latents": latents,
        "captions": captions,
        "image_paths": image_paths,
        "image_size": args.image_size,
        "split": args.split,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    print(f"Saved {latents.shape[0]} latents to: {output_path}")


if __name__ == "__main__":
    main()
