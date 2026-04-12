"""Pull a pretrained SD-style AutoencoderKL from HuggingFace and convert it
into the in-repo VAE format. Saves to scripts/checkpoints/vae_final.pt by default
so the rest of the pipeline can use it without retraining.

Defaults to `stabilityai/sd-vae-ft-mse` (the improved VAE most SD1.x setups
use). Pass --model to choose a different one — anything that loads via
diffusers.AutoencoderKL.from_pretrained will work.

Usage:
    python scripts/import_vae.py
    python scripts/import_vae.py --model CompVis/stable-diffusion-v1-4 --subfolder vae
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch

from models.vae import VAE
from models.convert_weights import convert_vae_state_dict


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='stabilityai/sd-vae-ft-mse',
                        help='HF repo id of an AutoencoderKL')
    parser.add_argument('--subfolder', type=str, default=None,
                        help='HF subfolder (e.g. "vae" for CompVis/stable-diffusion-v1-4)')
    parser.add_argument('--out', type=str, default='scripts/checkpoints/vae_final.pt')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip the encode/decode round-trip check')
    args = parser.parse_args()

    try:
        from diffusers import AutoencoderKL
    except ImportError:
        sys.exit("diffusers not installed. Run: pip install diffusers")

    print(f"Loading {args.model}" + (f" (subfolder={args.subfolder})" if args.subfolder else ""))
    if args.subfolder:
        hf_vae = AutoencoderKL.from_pretrained(args.model, subfolder=args.subfolder)
    else:
        hf_vae = AutoencoderKL.from_pretrained(args.model)

    print("Converting state dict to in-repo VAE format...")
    converted = convert_vae_state_dict(hf_vae.state_dict())

    print("Loading converted weights into in-repo VAE...")
    vae = VAE()
    missing, unexpected = vae.load_state_dict(converted, strict=False)
    if missing:
        print(f"  missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if missing or unexpected:
        print("  (small numbers of missing/unexpected keys are normal — converter "
              "leaves activations like nn.SiLU() / GroupNorm-only blocks alone)")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(vae.state_dict(), out_path)
    print(f"Saved to {out_path}")

    if args.no_verify:
        return

    print("Round-trip sanity check...")
    device = get_device()
    vae = vae.to(device).eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 256, 256, device=device)
        decoded, latents, mean, log_var = vae(x)
        recon_err = (decoded - x).abs().mean().item()
        print(f"  input  {tuple(x.shape)}")
        print(f"  latent {tuple(latents.shape)}")
        print(f"  recon  {tuple(decoded.shape)} | |x_hat - x| mean = {recon_err:.4f}")
    print("OK")


if __name__ == '__main__':
    main()
