import torch

from models.unet import UNet


def build_unet() -> UNet:
    return UNet(
        in_channels=4,
        out_channels=4,
        channels=64,
        n_res=1,
        channel_mults=(1, 2),
        attention_levels=(1,),
        n_heads=4,
        tf_layers=1,
        d_cond=512,
    )


def test_unet_unconditional_shape():
    model = build_unet().eval()
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([1, 2], dtype=torch.long)
    out = model(x, t)
    assert out.shape == x.shape


def test_unet_conditional_shape():
    model = build_unet().eval()
    x = torch.randn(2, 4, 16, 16)
    t = torch.tensor([3, 7], dtype=torch.long)
    cond = torch.randn(2, 77, 512)
    out = model(x, t, conditionings=cond)
    assert out.shape == x.shape
