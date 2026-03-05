"""Unet arch"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vae.VAE import AttentionBlock
from unet import TransformerBlock  # To Be Implemented


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int,  # input channel count
                 out_channels: int,  # output channel count
                 channels: int,  # operating channel count
                 n_res: int,  # res blocks per level
                 channel_mults: List[int],  # defines level count and channel expansion per layer
                 attention_levels: List[int],  # levels for attention
                 n_heads: int,  # attention block param
                 tf_layers: int = 1,  # transformer layers
                 d_cond: int = 768  # conditioning dimension
                 ):
        super().__init__()

        levels = len(channel_mults)
        channels_list = [channels * m for m in channel_mults]

        d_ts_embeddings = 4 * channels  # increasing emb dim seems arbitrary and imp dependent
        self.ts_embedding = nn.Sequential(
            nn.Linear(channels, d_ts_embeddings),
            nn.SiLU(),
            nn.Linear(d_ts_embeddings, d_ts_embeddings)
        )

        # Downblock
        self.down_blocks = nn.ModuleList()
        self.down_blocks.append(EmbeddingHandler(nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)))

        down_block_channels = [channels]

        for i in range(levels):
            for n in range(n_res):
                layers = [ResidualBlock(channels, d_ts_embeddings, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(AttentionBlock(channels, n_heads, tf_layers, d_cond))

                self.down_blocks.append(EmbeddingHandler(*layers))
                down_block_channels.append(channels)

            if i == levels - 1:
                self.down_blocks.append(EmbeddingHandler(DownSample(channels)))
                down_block_channels.append(channels)

        # Middleblock
        self.middle_blocks = EmbeddingHandler(
            ResidualBlock(channels, d_ts_embeddings),
            AttentionBlock(channels, n_heads, tf_layers, d_cond),
            ResidualBlock(channels, d_ts_embeddings),
        )

        # Upblock
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(levels)):
            for n in range(n_res + 1):
                layers = [
                    ResidualBlock(channels + down_block_channels.pop(), d_ts_embeddings, out_channels=channels_list[i])]
                channels = channels_list[i]

                if i in attention_levels:
                    layers.append(AttentionBlock(channels, n_heads, tf_layers, d_cond))

                if i != 0 and n == n_res:
                    layers.append(UpSample(channels))
                self.output_blocks.append(EmbeddingHandler(*layers))

        # Output layer
        self.out = nn.Sequential(
            TypedGroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, kernel_size=3, padding=1)
        )

    def generate_ts_signature(self, time_steps: torch.Tensor, max_period: int = 10000) -> torch.Tensor:
        half = self.channels // 2
        freq = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(time_steps.device)

        args = time_steps[:, None].float() * freq[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, conditionings: torch.Tensor) -> torch.Tensor:
        block_outputs = []

        ts_embeddings = self.generate_ts_signature(time_steps)
        ts_embeddings = self.ts_embedding(ts_embeddings)

        for module in self.down_blocks:
            x = module(x, ts_embeddings, conditionings)
            block_outputs.append(x)

        x = self.middle_blocks(x)

        for module in self.up_blocks:
            x = torch.cat([x, block_outputs.pop()], dim=1)  # append residuals
            x = module(x, ts_embeddings, conditionings)

        return self.out(x)


class UpSample(nn.Module):
    """Double h and w dimensions"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


class DownSample(nn.Module):
    """Halve h and w dimensions"""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, d_ts_embeddings: int, out_channels: int = None):
        super().__init__()

        if out_channels is None or out_channels == in_channels:
            out_channels = in_channels
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_ts_embeddings, out_channels)
        )

        self.in_layer = nn.Sequential(
            TypedGroupNorm(32, in_channels),  # Forces in_channels to be divisible by 32...
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        self.out_layer = nn.Sequential(
            TypedGroupNorm(32, in_channels),  # Forces in_channels to be divisible by 32...
            nn.SiLU(),
            nn.Dropout(0.),  # As per reference implementation...
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, ts_emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layer(x)
        ts_emb = self.emb_layer(ts_emb).type(h.dtype)
        h += ts_emb[:, :, None, None]
        h = self.out_layer(h)
        return self.skip(x) + h


class TypedGroupNorm(nn.GroupNorm):  # as per reference implementation
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class EmbeddingHandler(nn.Sequential):  # as per reference implementation
    def forward(self, x: torch.Tensor, ts_emb: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, ts_emb)
            elif isinstance(layer, TransformerBlock):
                x = layer(x, cond)
            else:
                x = layer(x)
        return x
