import torch
from models.clip import CLIPTextEncoder

def test_clip_smoke():
    model = CLIPTextEncoder.from_pretrained_hf().eval()
    with torch.no_grad():
        out = model.encode(["a photo of a dog", "a photo of a red car"])

    assert out.ndim == 3
    assert out.shape[0] == 2
    assert out.shape[1] == 77