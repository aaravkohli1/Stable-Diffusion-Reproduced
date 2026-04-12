"""Pre-compute VAE latents and CLIP text embeddings for UNet training.

Each cached file is a dict `{'latents': Tensor[4,h/8,w/8], 'conds': Tensor[T,D]}`
that `scripts/train_unet.py::CacheDataset` consumes directly.

Two source modes:
  --source local      DiffusionDataset (CIFAR-10 if --testing, else --root folder)
  --source hf         TrainingData (streams the ProGamerGov 1M synthetic shards)

Examples:

    # CIFAR-10 smoke test (50 samples)
    python scripts/compute_latents.py --vae-ckpt scripts/checkpoints/vae_final.pt \
        --source local --testing --num-samples 50 --image-size 256

    # Full SD-style cache from HF shards
    python scripts/compute_latents.py --vae-ckpt scripts/checkpoints/vae_final.pt \
        --source hf --shards 4 --max-samples 20000 --image-size 512 \
        --out-dir scripts/latents
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, IterableDataset

from data.dataset import DiffusionDataset, TrainingData
from models.vae import VAE
from models.clip import CLIPTextEncoder


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def collate(batch):
    images = torch.stack([b['image'] for b in batch], dim=0)
    captions = [b['caption'] for b in batch]
    return {'image': images, 'caption': captions}


def build_loader(args) -> DataLoader:
    if args.source == 'local':
        dataset = DiffusionDataset(
            testing=args.testing,
            num_test_samples=args.num_samples,
            image_size=args.image_size,
            root=args.root,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
    if args.source == 'hf':
        dataset = TrainingData(
            shards=args.shards,
            image_size=args.image_size,
            max_samples=args.max_samples,
        )
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=0,
            collate_fn=collate,
        )
    raise ValueError(f"unknown --source {args.source}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae-ckpt', type=str, required=True)
    parser.add_argument('--out-dir', type=str, default='scripts/latents')
    parser.add_argument('--source', type=str, default='local', choices=['local', 'hf'])
    parser.add_argument('--image-size', type=int, default=256)
    parser.add_argument('--batch-size', type=int, default=4)
    # local mode
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--testing', action='store_true',
                        help='[local] Use the CIFAR-10 smoke-test dataset')
    parser.add_argument('--root', type=str, default=None,
                        help='[local] Image folder root when --testing is not set')
    # hf mode
    parser.add_argument('--shards', type=int, default=1,
                        help='[hf] Number of webdataset shards to stream')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='[hf] Stop after this many samples')
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = build_loader(args)

    vae = VAE().to(device)
    state = torch.load(args.vae_ckpt, map_location=device, weights_only=True)
    vae.load_state_dict(state)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    text_encoder = CLIPTextEncoder.from_pretrained_hf().to(device)
    text_encoder.freeze()
    # Quick probe to confirm the CLIP hidden dim matches the UNet's d_cond
    with torch.no_grad():
        probe = text_encoder.encode(["test"])
        print(f"CLIP hidden dim: {probe.shape[-1]}")

    idx = 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            captions = list(batch['caption'])

            latents, _, _ = vae.encoder(images)  # returns (scaled_sample, mean, logvar)
            conds = text_encoder.encode(captions)

            latents = latents.cpu()
            conds = conds.cpu()

            for i in range(latents.shape[0]):
                torch.save(
                    {'latents': latents[i], 'conds': conds[i]},
                    out_dir / f"sample_{idx:06d}.pt",
                )
                idx += 1

            if idx % (args.batch_size * 10) == 0 or args.source == 'local':
                print(f"Cached {idx} samples")

    print(f"Wrote {idx} samples to {out_dir}")


if __name__ == '__main__':
    main()
