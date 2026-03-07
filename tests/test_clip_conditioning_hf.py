import torch

from models.clip import CLIPTextEncoder


def test_clip_encode_conditioning_hf():
    model = CLIPTextEncoder.from_pretrained_hf().eval()

    with torch.no_grad():
        cond, uncond = model.encode_conditioning(
            ["a photo of a dog", "a photo of a red car"],
            negative_texts=["blurry", "low quality"],
        )

    assert cond.ndim == 3
    assert uncond.ndim == 3
    assert cond.shape == uncond.shape
    assert cond.shape[0] == 2
    assert cond.shape[1] == 77
