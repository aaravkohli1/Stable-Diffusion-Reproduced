"""
Verify Flash Attention (SDPA) implementation in U-Net CrossAttention.
Compares F.scaled_dot_product_attention output vs manual attention computation.
"""
import torch
import torch.nn.functional as F


def _manual_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    """Baseline: manual softmax(QK^T/scale) @ V for comparison."""
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def test_sdpa_matches_manual_attention():
    """SDPA output should be numerically close to manual attention."""
    torch.manual_seed(42)
    b, n_heads, seq_q, seq_kv, head_dim = 2, 4, 64, 77, 32
    scale = head_dim ** -0.5

    q = torch.randn(b, n_heads, seq_q, head_dim)
    k = torch.randn(b, n_heads, seq_kv, head_dim)
    v = torch.randn(b, n_heads, seq_kv, head_dim)

    out_manual = _manual_attention(q, k, v, scale)
    out_sdpa = F.scaled_dot_product_attention(q, k, v, scale=scale)

    assert out_manual.shape == out_sdpa.shape
    # Allow small numerical differences (different backends may use different precision)
    assert torch.allclose(out_manual, out_sdpa, rtol=1e-2, atol=1e-2), (
        f"max diff: {(out_manual - out_sdpa).abs().max().item():.6f}"
    )


def test_cross_attention_forward():
    """CrossAttention with SDPA produces correct shape and no NaN."""
    from models.unet_attention import CrossAttention

    model = CrossAttention(query_dim=256, context_dim=512, n_heads=8)
    x = torch.randn(2, 64, 256)  # [B, N, C]
    context = torch.randn(2, 77, 512)  # [B, T, D]

    out = model(x, context)
    assert out.shape == x.shape
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_spatial_transformer_forward():
    """SpatialTransformer with SDPA-based attention produces correct shape."""
    from models.unet_attention import SpatialTransformer

    model = SpatialTransformer(channels=128, n_heads=4, tf_layers=1, d_cond=512)
    x = torch.randn(2, 128, 16, 16)  # [B, C, H, W]
    cond = torch.randn(2, 77, 512)

    out = model(x, cond)
    assert out.shape == x.shape
    assert not torch.isnan(out).any() and not torch.isinf(out).any()


def test_unet_with_sdpa_attention():
    """Full U-Net forward pass works with SDPA-based attention."""
    from models.unet import UNet

    model = UNet(
        in_channels=4,
        out_channels=4,
        channels=64,
        n_res=2,
        channel_mults=[1, 2, 4],
        attention_levels=[1, 2],
        n_heads=8,
        tf_layers=1,
        d_cond=512,
    )
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 77, 512)

    y = model(x, t, context)
    assert y.shape == x.shape
    assert not torch.isnan(y).any() and not torch.isinf(y).any()
