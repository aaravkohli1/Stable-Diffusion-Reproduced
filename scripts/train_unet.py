"""SD 1.x-style UNet training loop.

Reads a YAML config (see configs/standard.yaml). Implements:
  - bf16 / fp16 / fp32 mixed precision (bf16 default on H100)
  - AdamW + weight decay
  - linear LR warmup
  - gradient accumulation + gradient clipping
  - EMA copy of the weights (the artifact you actually sample from)
  - classifier-free-guidance training: random caption dropout, swapping conds for zeros
  - resume from the latest checkpoint via --resume
"""

import argparse
import copy
import math
import os
import signal
import sys
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from diffusion import (
    Diffuser,
    cosine_beta,
    linear_beta,
    quadratic_beta,
    scaled_linear_beta,
)
from models.unet import UNet


SCHEDULES = {
    "cosine": cosine_beta,
    "linear": linear_beta,
    "quadratic": quadratic_beta,
    "scaled_linear": scaled_linear_beta,
}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class CacheDataset(Dataset):
    """Loads pre-computed `{latents, conds}` tensors from a directory of .pt files."""

    def __init__(self, path: str):
        self.cache_dir = Path(path)
        self.latent_paths = sorted(self.cache_dir.glob('*.pt'))
        if not self.latent_paths:
            raise FileNotFoundError(f"No .pt latents found under {self.cache_dir}")

    def __len__(self) -> int:
        return len(self.latent_paths)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.latent_paths[idx], weights_only=True)


class EMA:
    """Maintains an exponential moving average of model parameters.

    SD-style training samples from the EMA copy, not the live one. The shadow
    can live on CPU (set ema_device: cpu in config) when GPU VRAM is tight.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999,
                 device: str | None = None):
        self.decay = decay
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.shadow = copy.deepcopy(model).eval().to(self.device)
        for p in self.shadow.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(
                p.detach().to(self.device, non_blocking=True),
                alpha=1 - self.decay,
            )
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b.to(self.device, non_blocking=True))


def lr_lambda(step: int, warmup: int) -> float:
    if warmup <= 0:
        return 1.0
    return min(1.0, (step + 1) / warmup)


PRECISION_DTYPES = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}


def find_latest_checkpoint(ckpt_dir: Path) -> Path | None:
    candidates = list(ckpt_dir.glob('unet_*.pt'))
    if not candidates:
        return None
    # Prefer numeric step ordering; fall back to mtime for unet_final.pt etc.
    def step_of(p: Path) -> int:
        try:
            return int(p.stem.split('_')[-1])
        except ValueError:
            return -1
    return max(candidates, key=step_of)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a checkpoint, or 'latest' to auto-pick the most recent")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # --- Distributed setup --------------------------------------------------
    use_ddp = 'RANK' in os.environ  # torchrun sets RANK, LOCAL_RANK, WORLD_SIZE
    if use_ddp:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        rank = 0
        world_size = 1
        device = get_device()

    is_main = rank == 0  # only rank 0 prints and saves

    if is_main:
        print(f"Using device: {device}" + (f" ({world_size} GPUs)" if use_ddp else ""))

    mcfg = config['model']
    tcfg = config['training']

    base_model = UNet(
        in_channels=mcfg["in_channels"],
        out_channels=mcfg['out_channels'],
        channels=mcfg['channels'],
        n_res=mcfg['n_res'],
        channel_mults=mcfg['channel_mults'],
        attention_levels=mcfg['attention_levels'],
        n_heads=mcfg['n_heads'],
        tf_layers=mcfg.get('tf_layers', 1),
        d_cond=mcfg.get('d_cond', 768),
    ).to(device)

    n_params = sum(p.numel() for p in base_model.parameters())
    if is_main:
        print(f"UNet params: {n_params/1e6:.1f}M")

    # torch.compile wraps the module — keep `base_model` as the canonical handle
    # for state-dict load/save and EMA, and use `model` for the training calls.
    compile_mode = tcfg.get('compile')  # None | "default" | "reduce-overhead" | "max-autotune"
    if compile_mode and device.type == 'cuda':
        if is_main:
            print(f"torch.compile mode={compile_mode}")
        model = torch.compile(base_model, mode=compile_mode)
    else:
        model = base_model

    # Wrap with DDP after compile
    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    dataset = CacheDataset(config['latents_path'])
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    dataloader = DataLoader(
        dataset,
        batch_size=tcfg['batch_size'],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=tcfg.get('num_workers', 0),
        pin_memory=(device.type == 'cuda'),
        drop_last=True,
    )
    if is_main:
        print(f"Dataset: {len(dataset)} cached samples, {len(dataloader)} batches/epoch")

    # Optimizer + EMA must reference the underlying parameters, not the
    # compiled wrapper, so they survive recompiles cleanly.
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=tcfg['lr'],
        betas=tuple(tcfg.get('betas', [0.9, 0.999])),
        weight_decay=tcfg.get('weight_decay', 1e-2),
    )

    warmup_steps = tcfg.get('warmup_steps', 0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, warmup_steps)
    )

    # Precision: bf16 (H100 default), fp16, or fp32. Backwards-compat: if the
    # config still has the old `mixed_precision: true`, treat as fp16.
    precision = tcfg.get('precision')
    if precision is None:
        precision = 'fp16' if tcfg.get('mixed_precision', True) else 'fp32'
    if precision not in PRECISION_DTYPES:
        raise ValueError(f"Unknown precision {precision!r}; choose fp32/fp16/bf16")
    amp_dtype = PRECISION_DTYPES[precision]
    use_amp = precision != 'fp32' and device.type == 'cuda'
    # GradScaler is only needed for fp16; bf16 has the dynamic range to skip it.
    scaler = torch.amp.GradScaler('cuda', enabled=(precision == 'fp16'))
    if is_main:
        print(f"Precision: {precision} (autocast={use_amp}, grad_scaler={scaler.is_enabled()})")

    grad_accum = max(1, tcfg.get('grad_accum_steps', 1))
    grad_clip = tcfg.get('grad_clip', 1.0)
    cfg_dropout = tcfg.get('cfg_dropout', 0.1)
    log_every = tcfg.get('log_interval', 50)
    save_every = tcfg.get('save_interval', 2000)
    max_steps = tcfg.get('max_steps', math.inf)

    env = Diffuser(timesteps=tcfg['diffusion_steps'], schedule=SCHEDULES[tcfg['beta_schedule']])

    use_ema = tcfg.get('ema', True)
    ema_device = tcfg.get('ema_device')  # default: same device as model
    # EMA only on rank 0 — no need to duplicate across GPUs.
    ema = EMA(base_model, decay=tcfg.get('ema_decay', 0.9999), device=ema_device) if (use_ema and is_main) else None

    checkpoints_dir = Path(__file__).resolve().parent / 'unet_checkpoints'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    optimizer.zero_grad(set_to_none=True)

    # Graceful shutdown on SIGINT/SIGTERM — save checkpoint before exiting.
    _stop_requested = False
    def _handle_stop(signum, frame):
        nonlocal _stop_requested
        if _stop_requested:
            sys.exit(1)  # second signal = hard exit
        _stop_requested = True
        print(f"\n[signal {signum}] Stopping after current step...")
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    # Resume
    if args.resume:
        if args.resume == 'latest':
            resume_path = find_latest_checkpoint(checkpoints_dir)
            if resume_path is None:
                if is_main:
                    print(f"--resume latest: no checkpoints in {checkpoints_dir}, starting fresh")
        else:
            resume_path = Path(args.resume)
        if resume_path is not None and resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            base_model.load_state_dict(ckpt['model'])
            if 'optimizer' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer'])
            if 'scaler' in ckpt and scaler.is_enabled():
                scaler.load_state_dict(ckpt['scaler'])
            if 'ema' in ckpt and ema is not None:
                ema.shadow.load_state_dict(ckpt['ema'])
            global_step = int(ckpt.get('step', 0))
            # Fast-forward the LR schedule. Compute the LR directly from the
            # config'd warmup function so we don't have to call scheduler.step()
            # before optimizer.step() (which PyTorch warns about).
            scheduler.last_epoch = global_step
            resumed_lr = tcfg['lr'] * lr_lambda(global_step - 1, warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = resumed_lr
            if is_main:
                print(f"Resumed from {resume_path} at step {global_step} (lr={resumed_lr:.2e})")

    for epoch in range(tcfg['epochs']):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for micro_step, batch in enumerate(dataloader):
            latents = batch['latents'].to(device, non_blocking=True)
            conds = batch['conds'].to(device, non_blocking=True)
            bsz = latents.shape[0]

            # Classifier-free guidance: drop captions to zero conditioning at random.
            if cfg_dropout > 0:
                drop_mask = torch.rand(bsz, device=device) < cfg_dropout
                if drop_mask.any():
                    conds = conds.clone()
                    conds[drop_mask] = 0.0

            t = torch.randint(0, tcfg['diffusion_steps'], (bsz,), device=device).long()

            with torch.amp.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
                loss = env.compute_loss(model, latents, t, conds=conds)
                loss = loss / grad_accum

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (micro_step + 1) % grad_accum == 0:
                if scaler.is_enabled():
                    if grad_clip:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip:
                        torch.nn.utils.clip_grad_norm_(base_model.parameters(), grad_clip)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(base_model)

                global_step += 1

                if is_main and global_step % log_every == 0:
                    cur_lr = scheduler.get_last_lr()[0]
                    print(
                        f"epoch {epoch+1} step {global_step} "
                        f"loss {loss.item()*grad_accum:.4f} lr {cur_lr:.2e}"
                    )

                if is_main and global_step % save_every == 0:
                    ckpt = {
                        'step': global_step,
                        'model': base_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scaler': scaler.state_dict(),
                    }
                    if ema is not None:
                        ckpt['ema'] = ema.shadow.state_dict()
                    path = checkpoints_dir / f"unet_{global_step}.pt"
                    torch.save(ckpt, path)
                    print(f"Checkpoint saved: {path}")

                if global_step >= max_steps or _stop_requested:
                    break

        if global_step >= max_steps or _stop_requested:
            break

    if is_main:
        final = {'step': global_step, 'model': base_model.state_dict()}
        if ema is not None:
            final['ema'] = ema.shadow.state_dict()
        final_path = checkpoints_dir / "unet_final.pt"
        torch.save(final, final_path)
        print(f"Final model saved: {final_path}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
