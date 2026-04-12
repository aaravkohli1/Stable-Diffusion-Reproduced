"""LoRA fine-tuning for the UNet on domain-specific data.

Loads pretrained CompVis SD1.4 weights (via diffusers), injects low-rank
adapters into the UNet's cross-attention Q/K/V/out projections, freezes
everything else, and trains only the LoRA parameters.

Usage:
    python scripts/finetune_lora.py configs/finetune.yaml
    python scripts/finetune_lora.py configs/finetune.yaml --resume latest
"""

import argparse
import math
import os
import signal
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from data.dataset import AxonometricDataset
from data.preprocessing import preprocess_image, preprocess_text
from diffusion import Diffuser, scaled_linear_beta, linear_beta, cosine_beta, quadratic_beta
from models.unet import UNet
from models.vae import VAE
from models.clip.text_encoder import CLIPTextEncoder

SCHEDULES = {
    "cosine": cosine_beta,
    "linear": linear_beta,
    "quadratic": quadratic_beta,
    "scaled_linear": scaled_linear_beta,
}


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear that adds a low-rank adapter.

    output = W_frozen @ x + (B @ A) @ x * (alpha / rank)

    Only A and B are trainable. The original weight W is frozen in place.
    """

    def __init__(self, original: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Freeze the original weight
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0,
                target_names: tuple[str, ...] = ('to_q', 'to_k', 'to_v', 'to_out')) -> list[str]:
    """Replace targeted nn.Linear modules with LoRALinear adapters.

    Returns the list of replaced module paths for logging.
    """
    replaced = []
    for name, module in model.named_modules():
        for attr in target_names:
            if hasattr(module, attr):
                original = getattr(module, attr)
                if isinstance(original, nn.Linear):
                    setattr(module, attr, LoRALinear(original, rank=rank, alpha=alpha))
                    replaced.append(f"{name}.{attr}")
    return replaced


def extract_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only the LoRA A/B parameters from the model."""
    return {k: v for k, v in model.state_dict().items()
            if 'lora_A' in k or 'lora_B' in k}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def find_latest_checkpoint(ckpt_dir: Path, prefix: str = 'lora_') -> Path | None:
    candidates = list(ckpt_dir.glob(f'{prefix}*.pt'))
    if not candidates:
        return None
    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except ValueError:
            return -1
    return max(candidates, key=step_of)


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


PRECISION_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def load_pretrained_sd14(vae: VAE, unet: UNet, repo_id: str = 'CompVis/stable-diffusion-v1-4'):
    """Download SD1.4 weights from HuggingFace and load into our models."""
    try:
        from diffusers import StableDiffusionPipeline
    except ImportError:
        sys.exit("diffusers not installed. Run: pip install diffusers transformers accelerate")

    from models.convert_weights import convert_vae_state_dict, convert_unet_state_dict

    print(f"Downloading {repo_id} from HuggingFace...")
    pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)

    vae_sd = convert_vae_state_dict(pipe.vae.state_dict())
    vae.load_state_dict(vae_sd, strict=True)

    unet_sd = convert_unet_state_dict(pipe.unet.state_dict())
    unet.load_state_dict(unet_sd, strict=True)

    # Build text encoder from the pipeline's own CLIP
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to a LoRA checkpoint, or 'latest'")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = get_device()
    print(f"Using device: {device}")

    mcfg = config['model']
    tcfg = config['training']
    lcfg = config['lora']

    # --- Build models -------------------------------------------------------
    unet = UNet(
        in_channels=mcfg['in_channels'],
        out_channels=mcfg['out_channels'],
        channels=mcfg['channels'],
        n_res=mcfg['n_res'],
        channel_mults=mcfg['channel_mults'],
        attention_levels=mcfg['attention_levels'],
        n_heads=mcfg['n_heads'],
        tf_layers=mcfg.get('tf_layers', 1),
        d_cond=mcfg.get('d_cond', 768),
    )
    vae = VAE()

    # Load pretrained weights
    pretrained = config.get('pretrained', 'CompVis/stable-diffusion-v1-4')
    text_encoder = load_pretrained_sd14(vae, unet, repo_id=pretrained)

    # Freeze VAE + text encoder (they don't train during LoRA fine-tuning)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False
    text_encoder = text_encoder.to(device)

    # Inject LoRA into UNet attention layers
    rank = lcfg.get('rank', 4)
    alpha = lcfg.get('alpha', 1.0)
    targets = tuple(lcfg.get('targets', ['to_q', 'to_k', 'to_v', 'to_out']))
    replaced = inject_lora(unet, rank=rank, alpha=alpha, target_names=targets)
    print(f"LoRA injected into {len(replaced)} layers (rank={rank}, alpha={alpha})")

    # Freeze everything in UNet except LoRA params
    for name, param in unet.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    unet = unet.to(device)
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"UNet: {total/1e6:.1f}M total, {trainable/1e6:.2f}M trainable ({100*trainable/total:.2f}%)")

    # --- Dataset ------------------------------------------------------------
    dataset = AxonometricDataset(
        root=config.get('data_root'),
        image_size=tcfg.get('image_size', 512),
    )

    def collate(batch):
        images = torch.stack([b['image'] for b in batch])
        captions = [b['caption'] for b in batch]
        return {'image': images, 'caption': captions}

    dataloader = DataLoader(
        dataset,
        batch_size=tcfg['batch_size'],
        shuffle=True,
        num_workers=tcfg.get('num_workers', 4),
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
        collate_fn=collate,
    )
    print(f"Dataset: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # --- Training setup -----------------------------------------------------
    precision = tcfg.get('precision', 'bf16')
    amp_dtype = PRECISION_DTYPES[precision]
    use_amp = precision != 'fp32' and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == 'fp16'))

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=tcfg['lr'],
        betas=tuple(tcfg.get('betas', [0.9, 0.999])),
        weight_decay=tcfg.get('weight_decay', 1e-2),
    )

    warmup_steps = tcfg.get('warmup_steps', 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, warmup_steps)
    )

    grad_accum = max(1, tcfg.get('grad_accum_steps', 1))
    grad_clip = tcfg.get('grad_clip', 1.0)
    cfg_dropout = tcfg.get('cfg_dropout', 0.1)
    log_every = tcfg.get('log_interval', 10)
    save_every = tcfg.get('save_interval', 500)
    max_steps = tcfg.get('max_steps', 5000)

    schedule_fn = SCHEDULES[tcfg.get('beta_schedule', 'scaled_linear')]
    diffusion_steps = tcfg.get('diffusion_steps', 1000)
    env = Diffuser(timesteps=diffusion_steps, schedule=schedule_fn)

    checkpoints_dir = Path(__file__).resolve().parent / 'lora_checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    # Graceful shutdown
    _stop_requested = False
    def _handle_stop(signum, frame):
        nonlocal _stop_requested
        if _stop_requested:
            sys.exit(1)
        _stop_requested = True
        print(f"\n[signal {signum}] Stopping after current step...")
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # Resume
    if args.resume:
        if args.resume == 'latest':
            resume_path = find_latest_checkpoint(checkpoints_dir)
            if resume_path is None:
                print("--resume latest: no checkpoints found, starting fresh")
        else:
            resume_path = Path(args.resume)
        if resume_path is not None and resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            # Load LoRA weights back into the model
            missing, unexpected = unet.load_state_dict(ckpt['lora'], strict=False)
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            global_step = int(ckpt.get('step', 0))
            scheduler.last_epoch = global_step
            resumed_lr = tcfg['lr'] * lr_lambda(global_step - 1, warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = resumed_lr
            print(f"Resumed from {resume_path} at step {global_step}")

    # --- Training loop ------------------------------------------------------
    print(f"\nTraining LoRA for {max_steps} steps...")

    for epoch in range(tcfg.get('epochs', 1000)):
        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            captions = batch['caption']

            # Encode images to latents (frozen VAE)
            with torch.no_grad():
                latents, _, _ = vae.encoder(images)

            # Encode text (frozen CLIP)
            with torch.no_grad():
                conds = text_encoder.encode(captions).to(device)

            bsz = latents.shape[0]

            # CFG dropout
            if cfg_dropout > 0:
                drop_mask = torch.rand(bsz, device=device) < cfg_dropout
                if drop_mask.any():
                    conds = conds.clone()
                    conds[drop_mask] = 0.0

            t = torch.randint(0, diffusion_steps, (bsz,), device=device).long()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = env.compute_loss(unet, latents, t, conds=conds)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (global_step + 1) % grad_accum == 0 or True:
                if scaler.is_enabled():
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(lora_params, grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            if global_step % log_every == 0:
                cur_lr = scheduler.get_last_lr()[0]
                print(f"epoch {epoch+1} step {global_step} "
                      f"loss {loss.item()*grad_accum:.4f} lr {cur_lr:.2e}")

            if global_step % save_every == 0:
                ckpt = {
                    'step': global_step,
                    'lora': extract_lora_state_dict(unet),
                    'optimizer': optimizer.state_dict(),
                    'config': config,
                }
                path = checkpoints_dir / f"lora_{global_step}.pt"
                torch.save(ckpt, path)
                print(f"Checkpoint saved: {path}")

            if global_step >= max_steps or _stop_requested:
                break

        if global_step >= max_steps or _stop_requested:
            break

    # Save final
    final = {
        'step': global_step,
        'lora': extract_lora_state_dict(unet),
        'config': config,
    }
    final_path = checkpoints_dir / 'lora_final.pt'
    torch.save(final, final_path)
    print(f"Final LoRA saved: {final_path}")


if __name__ == '__main__':
    main()
