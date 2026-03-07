"""Compact latent-space UNet with timestep and optional text conditioning."""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _num_groups(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def sinusoidal_timestep_embedding(
    timesteps: torch.Tensor,
    dim: int,
    max_period: int = 10000,
) -> torch.Tensor:
    """Create sinusoidal timestep embeddings of shape [B, dim]."""
    if timesteps.ndim == 0:
        timesteps = timesteps[None]

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / half
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb))[:, :, None, None].to(dtype=h.dtype)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CrossAttention2D(nn.Module):
    """Cross-attention over spatial tokens using CLIP token embeddings as context."""

    def __init__(self, channels: int, cond_dim: int, n_heads: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=n_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if cond is None:
            return x

        residual = x
        bsz, channels, height, width = x.shape
        x_seq = self.norm(x).reshape(bsz, channels, height * width).transpose(1, 2)  # [B, HW, C]

        if cond.ndim == 2:
            cond = cond[:, None, :]

        key_padding_mask = None
        if cond_mask is not None:
            if cond_mask.dtype != torch.bool:
                cond_mask = cond_mask > 0
            # PyTorch MHA expects True for positions to ignore.
            key_padding_mask = ~cond_mask

        attn_out, _ = self.attn(x_seq, cond, cond, key_padding_mask=key_padding_mask, need_weights=False)
        x_seq = x_seq + attn_out
        x_seq = x_seq + self.ff(self.ln(x_seq))

        x_out = x_seq.transpose(1, 2).reshape(bsz, channels, height, width)
        return residual + x_out


class ConditionedResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_dim: int,
        cond_dim: int,
        n_heads: int,
        with_attention: bool,
        tf_layers: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.res_block = ResidualBlock(in_channels, out_channels, time_dim=time_dim, dropout=dropout)
        if with_attention:
            self.attn_blocks = nn.ModuleList([CrossAttention2D(out_channels, cond_dim, n_heads) for _ in range(tf_layers)])
        else:
            self.attn_blocks = nn.ModuleList()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.res_block(x, time_emb)
        for attn in self.attn_blocks:
            x = attn(x, cond=cond, cond_mask=cond_mask)
        return x


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        channels: int = 128,
        n_res: int = 2,
        channel_mults: Iterable[int] = (1, 2, 4),
        attention_levels: Iterable[int] = (1, 2),
        n_heads: int = 8,
        tf_layers: int = 1,
        d_cond: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.n_res = n_res
        self.channel_mults = tuple(channel_mults)
        self.attention_levels = set(attention_levels)
        self.d_cond = d_cond

        time_dim = channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.input_conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)

        # Down path
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        down_skip_channels: list[int] = []
        ch = channels
        for level, mult in enumerate(self.channel_mults):
            out_ch = channels * mult
            level_blocks = nn.ModuleList()
            for _ in range(n_res):
                level_blocks.append(
                    ConditionedResBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        time_dim=time_dim,
                        cond_dim=d_cond,
                        n_heads=n_heads,
                        with_attention=level in self.attention_levels,
                        tf_layers=tf_layers,
                        dropout=dropout,
                    )
                )
                ch = out_ch
                down_skip_channels.append(ch)
            self.down_levels.append(level_blocks)
            if level < len(self.channel_mults) - 1:
                self.downsamples.append(Downsample(ch))

        # Middle path
        self.mid_block1 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)
        self.mid_attn = nn.ModuleList([CrossAttention2D(ch, d_cond, n_heads) for _ in range(tf_layers)])
        self.mid_block2 = ResidualBlock(ch, ch, time_dim=time_dim, dropout=dropout)

        # Up path
        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        up_skip_channels = down_skip_channels.copy()
        for level in reversed(range(len(self.channel_mults))):
            out_ch = channels * self.channel_mults[level]
            level_blocks = nn.ModuleList()
            for _ in range(n_res):
                skip_ch = up_skip_channels.pop()
                level_blocks.append(
                    ConditionedResBlock(
                        in_channels=ch + skip_ch,
                        out_channels=out_ch,
                        time_dim=time_dim,
                        cond_dim=d_cond,
                        n_heads=n_heads,
                        with_attention=level in self.attention_levels,
                        tf_layers=tf_layers,
                        dropout=dropout,
                    )
                )
                ch = out_ch
            self.up_levels.append(level_blocks)
            if level > 0:
                self.upsamples.append(Upsample(ch))

        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        time_steps: torch.Tensor,
        conditionings: torch.Tensor | None = None,
        condition_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if time_steps.ndim == 0:
            time_steps = time_steps.expand(x.shape[0])
        elif time_steps.shape[0] != x.shape[0]:
            if time_steps.numel() == 1:
                time_steps = time_steps.expand(x.shape[0])
            else:
                raise ValueError("time_steps batch size must match x batch size.")

        time_emb = sinusoidal_timestep_embedding(time_steps, self.channels)
        time_emb = self.time_mlp(time_emb)

        h = self.input_conv(x)
        skips: list[torch.Tensor] = []

        for level_idx, level_blocks in enumerate(self.down_levels):
            for block in level_blocks:
                h = block(h, time_emb=time_emb, cond=conditionings, cond_mask=condition_mask)
                skips.append(h)
            if level_idx < len(self.downsamples):
                h = self.downsamples[level_idx](h)

        h = self.mid_block1(h, time_emb=time_emb)
        for attn in self.mid_attn:
            h = attn(h, cond=conditionings, cond_mask=condition_mask)
        h = self.mid_block2(h, time_emb=time_emb)

        for level_idx, level_blocks in enumerate(self.up_levels):
            for block in level_blocks:
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)
                h = block(h, time_emb=time_emb, cond=conditionings, cond_mask=condition_mask)
            if level_idx < len(self.upsamples):
                h = self.upsamples[level_idx](h)

        return self.out_conv(F.silu(self.out_norm(h)))
