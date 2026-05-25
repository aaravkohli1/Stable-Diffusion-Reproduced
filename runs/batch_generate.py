"""Batch generation driver — loads the pipeline once and renders a list of prompts.

Used to produce the showcase grid on the project website.

Example:
    python -m runs.batch_generate --out docs/assets --from-pretrained CompVis/stable-diffusion-v1-4
"""

import argparse
import os
import sys

import torch

from models.vae import VAE
from models.unet import UNet
from models.clip.text_encoder import CLIPTextEncoder
from models.generate import (
    SD1_UNET_CONFIG, SCHEDULES, get_device, generate,
    load_from_pretrained, load_from_checkpoints,
)
from diffusion import Diffuser, linear_beta, scaled_linear_beta


SHOWCASE = [
    ("desert.png",     "a vast desert at golden hour, sweeping dunes, cinematic lighting"),
    ("dog.png",        "a corgi sitting in a meadow of wildflowers, soft focus, 50mm"),
    ("forest.png",     "an ancient misty pine forest, volumetric light, fine detail"),
    ("city.png",       "a neon-lit tokyo street at night, rainy pavement reflections"),
    ("mountain.png",   "alpine mountains at sunrise, alpenglow, sharp ridgelines, photoreal"),
    ("ocean.png",      "a crashing ocean wave, low angle, sun flare, hyperreal"),
    ("portrait.png",   "studio portrait of a young woman, rim light, kodak portra 400"),
    ("interior.png",   "a sunlit minimalist living room, scandinavian design, soft shadows"),
    ("astronaut.png",  "an astronaut on a pastel pink planet, surreal, cinematic"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="docs/assets")
    p.add_argument("--from-pretrained", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--vae-ckpt", default=None)
    p.add_argument("--unet-ckpt", default=None)
    p.add_argument("--sampler", default="dpm_pp_2m")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--karras", action="store_true")
    args = p.parse_args()

    device = get_device(args.device)
    print(f"Using device: {device}")
    os.makedirs(args.out, exist_ok=True)

    if args.from_pretrained:
        d_cond = 768
    else:
        d_cond = 512

    unet = UNet(**{**SD1_UNET_CONFIG, "d_cond": d_cond})
    vae = VAE()

    if args.from_pretrained:
        text_encoder = load_from_pretrained(args.from_pretrained, vae, unet, device)
        diffuser = Diffuser(1000, scaled_linear_beta)
    else:
        clip_name = "openai/clip-vit-base-patch32"
        text_encoder = CLIPTextEncoder.from_pretrained_hf(clip_name).freeze()
        load_from_checkpoints(args.vae_ckpt, args.unet_ckpt, vae, unet, device)
        diffuser = Diffuser(1000, linear_beta)

    unet = unet.to(device).eval()
    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device)

    for i, (name, prompt) in enumerate(SHOWCASE):
        torch.manual_seed(args.seed + i)
        print(f"\n[{i + 1}/{len(SHOWCASE)}] {name} :: {prompt}")
        img = generate(
            prompt=prompt,
            text_encoder=text_encoder,
            unet=unet,
            vae=vae,
            diffuser=diffuser,
            device=device,
            sampler=args.sampler,
            num_steps=args.steps,
            height=args.size,
            width=args.size,
            guidance_scale=args.guidance_scale,
            use_karras=args.karras,
        )
        path = os.path.join(args.out, name)
        img.save(path)
        print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
