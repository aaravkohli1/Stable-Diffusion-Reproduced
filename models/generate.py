"""Text to image generation script.

Example:
    With pretrained SD 1.4 weights from HuggingFace:
    python3 -m models.generate --prompt "a photo of a cat" --from-pretrained CompVis/stable-diffusion-v1-4 --karras

Local checkpoints are supported, but not available yet
"""

import argparse
import sys

import torch
from PIL import Image

from models.vae import VAE
from models.unet import UNet
from models.clip.text_encoder import CLIPTextEncoder
from models.convert_weights import convert_vae_state_dict, convert_unet_state_dict
from diffusion import Diffuser, linear_beta, cosine_beta, quadratic_beta, scaled_linear_beta
from diffusion.sampling import (
    step_probabilistic, sample_euler_ancestral,
    sample_ddim, sample_heun, sample_dpm_pp_2m,
)

SD1_UNET_CONFIG = dict(
    in_channels=4,
    out_channels=4,
    channels=320,
    n_res=2,
    channel_mults=[1, 2, 4, 4],
    attention_levels=[0, 1, 2],
    n_heads=8,
    tf_layers=1,
    d_cond=768,
)

SCHEDULES = {
    'linear': linear_beta,
    'cosine': cosine_beta,
    'quadratic': quadratic_beta,
    'scaled_linear': scaled_linear_beta,
}


def get_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_from_pretrained(repo_id, vae, unet, device):
    """Download a diffusers SD pipeline and convert weights into our models."""
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        print('Error: diffusers not installed')
        print('Install it with: pip install diffusers transformers accelerate')
        sys.exit(1)

    print(f'Downloading {repo_id} from HF')
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32)

    print('Converting VAE weights...')
    vae_sd = convert_vae_state_dict(pipe.vae.state_dict())
    missing, unexpected = vae.load_state_dict(vae_sd, strict=True)
    if missing:
        print(f'  VAE missing keys: {missing}')
    if unexpected:
        print(f'  VAE unexpected keys: {unexpected}')

    print('Converting UNet weights ')
    unet_sd = convert_unet_state_dict(pipe.unet.state_dict())
    missing, unexpected = unet.load_state_dict(unet_sd, strict=True)
    if missing:
        print(f'  UNet missing keys: {missing}')
    if unexpected:
        print(f'  UNet unexpected keys: {unexpected}')

    print('Loading CLIP text encoder')
    from models.clip.tokenizer import CLIPTokenizerWrapper
    from models.clip.clip_text_model import MyCLIPTextModel

    hf_te = pipe.text_encoder
    cfg = hf_te.config
    my_clip = MyCLIPTextModel(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        max_position_embeddings=cfg.max_position_embeddings,
        layer_norm_eps=cfg.layer_norm_eps,
    )
    my_clip.load_state_dict(hf_te.state_dict(), strict=True)
    my_clip.eval()

    tokenizer = CLIPTokenizerWrapper(pipe.tokenizer, max_length=77)
    text_encoder = CLIPTextEncoder(model=my_clip, tokenizer=tokenizer).freeze()

    del pipe
    return text_encoder


def load_from_checkpoints(vae_ckpt, unet_ckpt, vae, unet, device):
    """Load our own trained checkpoints."""
    if vae_ckpt:
        print(f'Loading VAE from {vae_ckpt}...')
        vae.load_state_dict(torch.load(vae_ckpt, map_location=device, weights_only=True))
    if unet_ckpt:
        print(f'Loading UNet from {unet_ckpt}...')
        unet.load_state_dict(torch.load(unet_ckpt, map_location=device, weights_only=True))


@torch.no_grad()
def generate(
    prompt,
    text_encoder,
    unet,
    vae,
    diffuser,
    device,
    sampler='euler_a',
    num_steps=20,
    height=512,
    width=512,
    guidance_scale=7.5,
    use_karras=False,
    cfg_rescale=0.0,
):
    """Run the full text-to-image pipeline."""
    latent_h, latent_w = height // 8, width // 8
    shape = (1, 4, latent_h, latent_w)

    kwargs = dict(
        textencoder=text_encoder, prompts=[prompt], guidance_scale=guidance_scale,
        cfg_rescale=cfg_rescale,
    )

    karras_tag = ' +karras' if use_karras else ''

    if sampler == 'euler_a':
        print(f'Sampling with Euler Ancestral ({num_steps} steps{karras_tag})...')
        latents = sample_euler_ancestral(unet, diffuser, shape, num_steps=num_steps,
                                         use_karras=use_karras, **kwargs)
    elif sampler == 'ddim':
        print(f'Sampling with DDIM ({num_steps} steps, eta=0)...')
        latents = sample_ddim(unet, diffuser, shape, num_steps=num_steps, eta=0.0, **kwargs)
    elif sampler == 'heun':
        print(f'Sampling with Heun ({num_steps} steps{karras_tag}, 2× evals)...')
        latents = sample_heun(unet, diffuser, shape, num_steps=num_steps,
                              use_karras=use_karras, **kwargs)
    elif sampler == 'dpm_pp_2m':
        print(f'Sampling with DPM-Solver++ 2M ({num_steps} steps{karras_tag})...')
        latents = sample_dpm_pp_2m(unet, diffuser, shape, num_steps=num_steps,
                                    use_karras=use_karras, **kwargs)
    else:
        # DDPM
        cond  = text_encoder.encode([prompt]).to(device)
        uncond = text_encoder.encode(['']).to(device)
        latents = torch.randn(shape, device=device)

        timesteps = list(range(diffuser.timesteps))[::-1]
        print(f'Sampling with DDPM ({diffuser.timesteps} steps)...')
        for i, t in enumerate(timesteps):
            latents = step_probabilistic(
                unet, diffuser, latents, t,
                cond_context=cond,
                uncond_context=uncond,
                guidance_scale=guidance_scale,
            )
            if (i + 1) % 100 == 0 or i == len(timesteps) - 1:
                print(f'  Step {i + 1}/{len(timesteps)}')

    print('Decoding latents')
    image = vae.decode(latents)
    image = image.clamp(-1, 1)
    image = (image + 1) / 2 
    image = image.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
    image = (image * 255).clip(0, 255).astype('uint8')
    return Image.fromarray(image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='UTMIST / Stable Diffusion')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt')
    parser.add_argument('--output', type=str, default='output.png', help='Output image path')
    parser.add_argument('--from-pretrained', type=str, default=None,
                        help='HuggingFace model ID (e.g. CompVis/stable-diffusion-v1-4)')
    parser.add_argument('--vae-ckpt', type=str, default=None, help='Path to VAE checkpoint')
    parser.add_argument('--unet-ckpt', type=str, default=None, help='Path to UNet checkpoint')
    parser.add_argument('--sampler', type=str, default='dpm_pp_2m',
                        choices=['euler_a', 'ddim', 'heun', 'dpm_pp_2m', 'ddpm'],
                        help='Sampling method (default: dpm_pp_2m)')
    parser.add_argument('--steps', type=int, default=None, help='Sampling steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5, help='CFG scale')
    parser.add_argument('--height', type=int, default=512, help='Image height (multiple of 8)')
    parser.add_argument('--width', type=int, default=512, help='Image width (multiple of 8)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu/mps)')
    parser.add_argument('--schedule', type=str, default=None,
                        choices=['linear', 'cosine', 'quadratic', 'scaled_linear'],
                        help='Noise schedule (default: scaled_linear for pretrained, linear otherwise)')
    parser.add_argument('--karras', action='store_true',
                        help='Use Karras sigma schedule (more steps at low noise for finer details)')
    parser.add_argument('--cfg-rescale', type=float, default=0.0,
                        help='CFG rescale factor (0.0=off, 0.7=recommended). Reduces over-saturation.')
    args = parser.parse_args()

    device = get_device(args.device)
    print(f'Using device: {device}')

    if args.seed is not None:
        torch.manual_seed(args.seed)

    if args.from_pretrained:
        clip_name = 'openai/clip-vit-large-patch14'
        d_cond = 768
    else:
        clip_name = 'openai/clip-vit-base-patch32'
        d_cond = 512

    unet_config = {**SD1_UNET_CONFIG, 'd_cond': d_cond}
    unet = UNet(**unet_config)
    vae = VAE()

    if args.from_pretrained:
        text_encoder = load_from_pretrained(args.from_pretrained, vae, unet, device)
    else:
        text_encoder = CLIPTextEncoder.from_pretrained_hf(clip_name).freeze()
        load_from_checkpoints(args.vae_ckpt, args.unet_ckpt, vae, unet, device)

    unet = unet.to(device).eval()
    vae = vae.to(device).eval()
    text_encoder = text_encoder.to(device)

    if args.schedule:
        schedule_fn = SCHEDULES[args.schedule]
    elif args.from_pretrained:
        schedule_fn = scaled_linear_beta
    else:
        schedule_fn = linear_beta
    diffuser = Diffuser(1000, schedule_fn)

    _default_steps = {'euler_a': 20, 'heun': 20, 'dpm_pp_2m': 20, 'ddim': 50, 'ddpm': 1000}
    num_steps = args.steps if args.steps is not None else _default_steps[args.sampler]

    image = generate(
        prompt=args.prompt,
        text_encoder=text_encoder,
        unet=unet,
        vae=vae,
        diffuser=diffuser,
        device=device,
        sampler=args.sampler,
        num_steps=num_steps,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        use_karras=args.karras,
        cfg_rescale=args.cfg_rescale,
    )

    image.save(args.output)
    print(f'Saved to {args.output}')
