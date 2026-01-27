"""Image and text preprocessing utilities"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Union

def preprocess_image(
        image: Union[Image.Image, np.ndarray], 
        image_size: int = 256
        ) -> torch.Tensor:
    """Preprocess image for VAE, returns normalized image tensor [3, H, W] in [-1, 1]"""

    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    return transform(image)

def preprocess_text(text: str) -> str:
    """Lowercase, strip, and normalize whitespace in caption text."""
    return ' '.join(text.lower().strip().split())


