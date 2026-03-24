import torch
from models.unet import UNet

def build_model():
    return UNet(
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

def test_unet_forward_shape_with_context():
    model = build_model()
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 77, 512)

    y = model(x, t, context)
    assert y.shape == x.shape

def test_unet_forward_shape_without_context():
    model = build_model()
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))

    y = model(x, t, None)
    assert y.shape == x.shape

def test_unet_conditioning_changes_output():
    model = build_model()
    x = torch.randn(1, 4, 32, 32)
    t = torch.randint(0, 1000, (1,))
    c1 = torch.randn(1, 77, 512)
    c2 = torch.randn(1, 77, 512)

    y1 = model(x, t, c1)
    y2 = model(x, t, c2)

    assert not torch.allclose(y1, y2)