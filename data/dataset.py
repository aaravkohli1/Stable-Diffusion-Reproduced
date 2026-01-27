"""Dataset classes for SD training"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from .preprocessing import preprocess_image, preprocess_text
from typing import Union

class DiffusionDataset(Dataset):
    """Dataset for VAE and CLIP"""

    def __init__(self, path: str, image_size: int = 256, split: str = 'train'):
        """
        Args:
            path: Path to dataset directory
            image_size: Train image size (Square)
            split: Train or Validation ('validation')
        """
        self.path = Path(path)
        self.image_size = image_size
        self.split = split
        self.image_dir = self.path / split
        self.image_paths = sorted(
            list(self.image_dir.glob('*.jpg')) +
            list(self.image_dir.glob('*.png'))
            )

        caption_file = self.path / split / 'captions.json'
        if caption_file.exists():
            with open(caption_file, 'r') as f:
                self.captions = json.load(f)
        else:
            raise FileNotFoundError(
                "captions.json: File Not Found :(\n" 
                "Expects captions in your train/validation folder"
                )
        
    def __len__(self) -> int:
        """Return the number of images"""
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        """Retrieve the image at a specific index"""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = preprocess_image(image, self.image_size)
        caption = self.captions.get(image_path.name, '')
        caption = preprocess_text(caption)

        return {
            'image': image,
            'caption': caption,
            'image_path': str(image_path)
            }
    