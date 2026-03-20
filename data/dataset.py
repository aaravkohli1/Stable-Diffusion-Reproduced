"""Dataset classes for SD training"""

import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import json
from .preprocessing import preprocess_image, preprocess_text
from typing import Union
import torchvision.datasets as datasets
import requests
from io import BytesIO
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

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
            testing: If True, use CIFAR-10 for quick testing
            num_test_samples: Number of samples to use (test mode or LAION subset)
        """

        if testing:
            self.__init__test_data(image_size, num_test_samples)
            return

        self.__init__laion_data(image_size, num_test_samples)

    def __init__laion_data(self, image_size: int, num_samples: int):
        """Download a subset of LAION Aesthetics from HuggingFace"""
        from datasets import load_dataset

        self.test_mode = False
        self.laion_mode = True
        self.image_size = image_size

        cache_dir = Path(__file__).resolve().parent / 'laion_aesthetic_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)

        images_dir = cache_dir / 'images'
        images_dir.mkdir(exist_ok=True)
        captions_file = cache_dir / 'captions.json'

        # Check if we already have enough cached samples
        if captions_file.exists():
            with open(captions_file, 'r') as f:
                cached = json.load(f)
            existing = [
                p for p in sorted(images_dir.glob('*.jpg'))
                if p.name in cached
            ]
            if len(existing) >= num_samples:
                self.image_paths = existing[:num_samples]
                self.captions = cached
                return

        # Stream LAION aesthetic dataset metadata
        hf_token = os.getenv('HF_TOKEN')
        ds = load_dataset(
            'laion/laion2B-en-aesthetic',
            split='train',
            streaming=True,
            token=hf_token
        )

        captions = {}
        if captions_file.exists():
            with open(captions_file, 'r') as f:
                captions = json.load(f)

        downloaded = len(list(images_dir.glob('*.jpg')))
        for i, sample in enumerate(ds):
            if downloaded >= num_samples:
                break

            fname = f"{downloaded:06d}.jpg"
            fpath = images_dir / fname

            try:
                resp = requests.get(sample['URL'], timeout=5)
                resp.raise_for_status()
                img = Image.open(BytesIO(resp.content)).convert('RGB')
                img.save(fpath)
                captions[fname] = sample['TEXT']
                downloaded += 1
            except Exception as e:
                print(f"[LAION] Skipping sample {i}: {e}")
                continue

        with open(captions_file, 'w') as f:
            json.dump(captions, f)

        self.image_paths = sorted(images_dir.glob('*.jpg'))[:num_samples]
        self.captions = captions
        
        
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
            'image_path': str(image_path),
            }
    