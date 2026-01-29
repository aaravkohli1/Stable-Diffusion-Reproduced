"""Dataset classes for SD training"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from .preprocessing import preprocess_image, preprocess_text
from typing import Union
import torchvision.datasets as datasets 

class DiffusionDataset(Dataset):
    """Dataset for VAE and CLIP"""

    def __init__(
            self, 
            path: str = '', 
            image_size: int = 256, 
            split: str = 'train',
            testing: bool = False,
            num_test_samples: int = 20
            ):
        """
        Args:
            path: Path to dataset directory
            image_size: Train image size (Square)
            split: Train or Validation ('validation')
        """

        if testing:
            self.__init__test_data(image_size, num_test_samples)
            return

        self.test_mode = False
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
        if self.test_mode:
            return len(self.indices)
        return len(self.image_paths)
    

    def __init__test_data(self, image_size: int, num_test_samples: int):
        """Retrieve CIFAR Images for Testing"""
        self.test_mode = True
        self.image_size = image_size
        
        self.dataset = datasets.CIFAR10(
            root='data/cifar10',
            train=True,
            download=True
        )
        
        self.indices = list(range(num_test_samples))
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
  
    def __getitem__(self, idx: int) -> dict[str, Union[torch.Tensor, str]]:
        """Retrieve the image at a specific index"""

        if self.test_mode:
            image, label = self.dataset[self.indices[idx]]
            image = preprocess_image(image, self.image_size)
            caption = f"a photo of a {self.class_names[label]}"
            
            return {
                'image': image,
                'caption': caption,
                'image_path': f"cifar10_{idx}"
            }
        
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
    