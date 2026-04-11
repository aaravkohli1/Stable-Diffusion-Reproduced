"""
Attention maps
See test file for how to use this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image


def _is_cross_attention(module):
    """Check if a module is a CrossAttention instance without importing the class."""
    return type(module).__name__ == 'CrossAttention'


class AttentionMapStore:
    """Accumulates cross-attention maps during UNet forward passes."""

    def __init__(self):
        self.maps: Dict[str, List[torch.Tensor]] = {}

    def clear(self):
        self.maps.clear()

    def record(self, name: str, attn_weights: torch.Tensor):
        """Record attention weights. attn_weights: [B, heads, N, T]."""
        if name not in self.maps:
            self.maps[name] = []
        self.maps[name].append(attn_weights.detach().cpu())

    def get_layers_by_resolution(self, spatial_n: Optional[int] = None) -> List[str]:
        """Return layer names matching a spatial resolution (N = H*W).

        If spatial_n is None, returns all layers.
        """
        if spatial_n is None:
            return list(self.maps.keys())
        return [
            name for name, maps in self.maps.items()
            if maps[0].shape[2] == spatial_n
        ]

    def aggregate(
        self,
        layer_name: Optional[str] = None,
        layer_names: Optional[List[str]] = None,
        head_reduction: str = 'mean',
        timestep: int = -1,
        timestep_range: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return aggregated attention map [B, N, T]."""
        if layer_names is not None:
            names = layer_names
        elif layer_name is not None:
            names = [layer_name]
        else:
            names = list(self.maps.keys())

        if timestep_range is not None:
            t_start, t_end = timestep_range
        else:
            total_t = len(next(iter(self.maps.values())))
            t_idx = timestep if timestep >= 0 else total_t + timestep
            t_start, t_end = t_idx, t_idx + 1

        all_maps = []
        target_n = self.maps[names[0]][0].shape[2]

        for name in names:
            for t in range(t_start, t_end):
                m = self.maps[name][t]
                if m.shape[2] != target_n:
                    b, h, n, tok = m.shape
                    m = m.reshape(b * h, 1, n, tok)
                    m = F.interpolate(m, size=(target_n, tok),
                                      mode='bilinear', align_corners=False)
                    m = m.reshape(b, h, target_n, tok)
                all_maps.append(m)

        attn = torch.stack(all_maps).mean(dim=0)  # [B, heads, N, T]

        if head_reduction == 'mean':
            return attn.mean(dim=1)
        elif head_reduction == 'max':
            return attn.max(dim=1).values
        else:
            raise ValueError(f'Unknown head_reduction: {head_reduction}')

    def get_spatial_map(
        self,
        token_index: int,
        spatial_resolution: Tuple[int, int],
        layer_name: Optional[str] = None,
        head_reduction: str = 'mean',
        timestep: int = -1,
        batch_index: int = 0,
    ) -> np.ndarray:
        """Return [H, W] attention heatmap for a single token."""
        agg = self.aggregate(layer_name, head_reduction=head_reduction,
                             timestep=timestep)
        token_map = agg[batch_index, :, token_index]
        h, w = spatial_resolution
        return token_map.reshape(h, w).numpy()

    @property
    def layer_names(self) -> List[str]:
        return list(self.maps.keys())

    @property
    def num_timesteps(self) -> int:
        if not self.maps:
            return 0
        return len(next(iter(self.maps.values())))


def _make_cross_attn_hook(store: AttentionMapStore, name: str):
    """Create a forward hook that computes attention weights alongside the fused kernel."""

    def hook(module, inputs, output):
        x = inputs[0]       # [B, N, C]  (may be 2*B during CFG)
        context = inputs[1]  # [B, T, D]

        with torch.no_grad():
            b, n, _ = x.shape
            _, t, _ = context.shape

            q = module.to_q(x)
            k = module.to_k(context)

            q = q.view(b, n, module.n_heads, module.head_dim).transpose(1, 2)
            k = k.view(b, t, module.n_heads, module.head_dim).transpose(1, 2)

            # [B, heads, N, T]
            attn_weights = torch.softmax(
                torch.matmul(q, k.transpose(-1, -2)) * module.scale,
                dim=-1,
            )

            # During CFG, batch is doubled [uncond, cond]. Keep only cond half.
            if b > 1 and b % 2 == 0:
                attn_weights = attn_weights[b // 2:]

            store.record(name, attn_weights)

    return hook


@contextmanager
def capture_attention(unet: torch.nn.Module):
    """Context manager that captures cross-attention maps during inference."""
    store = AttentionMapStore()
    hooks = []

    for name, module in unet.named_modules():
        if _is_cross_attention(module) and 'cross_attn' in name:
            h = module.register_forward_hook(_make_cross_attn_hook(store, name))
            hooks.append(h)

    try:
        yield store
    finally:
        for h in hooks:
            h.remove()


def _select_mid_layers(store: AttentionMapStore) -> List[str]:
    """Pick the mid-resolution layers (not the largest, not the smallest)."""
    resolutions = {}
    for name, maps in store.maps.items():
        n = maps[0].shape[2]
        resolutions.setdefault(n, []).append(name)

    sorted_res = sorted(resolutions.keys())

    if 256 in resolutions:
        return resolutions[256]
    if len(sorted_res) >= 3:
        return resolutions[sorted_res[1]]
    return resolutions[sorted_res[0]]


def visualize_attention(
    store: AttentionMapStore,
    tokens: List[str],
    image: Union[Image.Image, np.ndarray],
    token_indices: Optional[List[int]] = None,
    timestep: int = -1,
    timestep_aggregate: bool = True,
    head_reduction: str = 'mean',
    layer_name: Optional[str] = None,
    save_path: Optional[str] = None,
):
    """Overlay per-token attention heatmaps on the generated image."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    if token_indices is None:
        token_indices = list(range(len(tokens)))

    if layer_name is not None:
        layers = [layer_name]
    else:
        layers = _select_mid_layers(store)

    total_t = store.num_timesteps
    if timestep_aggregate and total_t > 1:
        t_range = (total_t // 2, total_t)
    else:
        t_idx = timestep if timestep >= 0 else total_t + timestep
        t_range = (t_idx, t_idx + 1)

    agg = store.aggregate(
        layer_names=layers,
        head_reduction=head_reduction,
        timestep_range=t_range,
    ) 

    n = agg.shape[1]
    spatial_side = int(n ** 0.5)
    img_h, img_w = image.shape[:2]

    n_tokens = len(token_indices)
    cols = min(n_tokens, 5)
    rows = (n_tokens + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, tok_idx in enumerate(token_indices):
        r, c = divmod(idx, cols)
        ax = axes[r, c]

        heatmap = agg[0, :, tok_idx].reshape(1, 1, spatial_side, spatial_side)
        heatmap = F.interpolate(
            heatmap, size=(img_h, img_w), mode="bicubic", align_corners=False
        )
        heatmap = heatmap.squeeze().numpy()

        lo = np.percentile(heatmap, 2)
        hi = np.percentile(heatmap, 98)
        heatmap = np.clip((heatmap - lo) / (hi - lo + 1e-8), 0, 1)

        heatmap = heatmap ** 1.5

        ax.imshow(image)
        ax.imshow(heatmap, cmap='inferno', alpha=0.6, vmin=0, vmax=1)
        label = tokens[idx] if idx < len(tokens) else f'[{tok_idx}]'
        ax.set_title(f'{label}', fontsize=12, fontweight='bold')
        ax.axis('off')

    for idx in range(n_tokens, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
