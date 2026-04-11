from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, n_heads: int):
        super().__init__()
        assert query_dim % n_heads == 0, "query_dim must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        x: [B, N, C]
        context: [B, T, D]
        """
        b, n, c = x.shape
        _, t, _ = context.shape

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q = q.view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(b, n, c)
        return self.to_out(out)


class GEGLU(nn.Module):
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        hidden = dim * mult
        self.geglu = GEGLU(dim, hidden)
        self.linear = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.geglu(x))


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, d_cond: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = CrossAttention(dim, dim, n_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, d_cond, n_heads)

        self.norm3 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.self_attn(self.norm1(x), self.norm1(x))
        if cond is not None:
            x = x + self.cross_attn(self.norm2(x), cond)
        x = x + self.ff(self.norm3(x))
        return x


class SpatialTransformer(nn.Module):
    def __init__(self, channels: int, n_heads: int, tf_layers: int, d_cond: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.proj_in = nn.Conv2d(channels, channels, kernel_size=1)

        self.blocks = nn.ModuleList([
            BasicTransformerBlock(channels, n_heads, d_cond)
            for _ in range(tf_layers)
        ])

        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: [B, C, H, W]
        cond: [B, 77, D_cond] or None
        """
        b, c, h, w = x.shape
        residual = x

        x = self.proj_in(self.norm(x))
        x = x.permute(0, 2, 3, 1).reshape(b, h * w, c) 

        for block in self.blocks:
            x = block(x, cond)

        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        x = self.proj_out(x)

        return x + residual