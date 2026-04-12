"""Datasets for training and fine tuning"""

from pathlib import Path
from typing import Optional

from torch.utils.data import Dataset, IterableDataset
from datasets import load_dataset
from torchvision.datasets import CIFAR10

from .preprocessing import preprocess_image, preprocess_text


class TrainingData(IterableDataset):
    """Streams the ProGamerGov synthetic-1M sharded webdataset from HuggingFace.

    Captions live under sample['json']['short_caption']; the .tar files contain
    .jpg + .json pairs (no .txt sidecar). `image_size` defaults to 512 to match
    SD 1.x training resolution.
    """

    BASE_URL = 'https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions/resolve/main/data/data-{i:06d}.tar'

    def __init__(
        self,
        shards: int,
        image_size: int = 512,
        caption_key: str = 'short_caption',
        max_samples: Optional[int] = None,
    ):
        urls = [self.BASE_URL.format(i=i) for i in range(shards)]
        self.dataset = load_dataset(
            'webdataset', data_files={'train': urls}, split='train', streaming=True
        )
        self.image_size = image_size
        self.caption_key = caption_key
        self.max_samples = max_samples

    def __iter__(self):
        n = 0
        for sample in self.dataset:
            meta = sample.get('json') or {}
            caption = meta.get(self.caption_key) or meta.get('short_caption') \
                or meta.get('long_caption') or ''
            image = sample.get('jpg')
            if image is None or not caption:
                continue
            yield {
                'image': preprocess_image(image, image_size=self.image_size),
                'caption': preprocess_text(caption),
            }
            n += 1
            if self.max_samples is not None and n >= self.max_samples:
                return


class DiffusionDataset(Dataset):
    """Map-style dataset used by the VAE/UNet training scripts.

    In `testing=True` mode it wraps CIFAR-10 (downloaded on first use) so the
    pipeline can be smoke-tested without external data. Otherwise it walks a
    local image folder of jpg/png files.

    Each item is `{'image': Tensor[C,H,W] in [-1,1], 'caption': str}`.
    """

    _IMG_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')

    def __init__(
        self,
        testing: bool = False,
        num_test_samples: int = 100,
        image_size: int = 256,
        root: Optional[str] = None,
    ) -> None:
        self.image_size = image_size
        self.testing = testing

        if testing:
            cifar_root = root or str(Path(__file__).resolve().parent / '_cifar10')
            self._cifar = CIFAR10(root=cifar_root, train=True, download=True)
            self._length = min(num_test_samples, len(self._cifar))
            self._paths = None
        else:
            if root is None:
                root = str(Path(__file__).resolve().parent / 'images')
            root_path = Path(root)
            if not root_path.exists():
                raise FileNotFoundError(
                    f"DiffusionDataset(testing=False) expected images at {root_path}. "
                    f"Pass `root=` or set testing=True for a CIFAR-10 smoke test."
                )
            self._paths = sorted(
                p for p in root_path.rglob('*') if p.suffix.lower() in self._IMG_EXTS
            )
            if not self._paths:
                raise FileNotFoundError(f"No images found under {root_path}")
            self._length = len(self._paths)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict:
        if self.testing:
            pil_image, label = self._cifar[idx]
            caption = self._cifar.classes[label]
        else:
            from PIL import Image
            path = self._paths[idx]
            pil_image = Image.open(path).convert('RGB')
            caption = path.stem.replace('_', ' ')

        return {
            'image': preprocess_image(pil_image, image_size=self.image_size),
            'caption': preprocess_text(caption),
        }


class AxonometricDataset(Dataset):
    """Dataset for axonometric / isometric image-caption pairs.

    Intended for fine-tuning experiments on spatially-aware imagery
    (isometric architecture, axonometric renders, etc.).

    Implementation should:
      1. Load or stream image-caption pairs from a suitable source
         (e.g. DiffusionDB filtered for isometric prompts, or a
         curated local folder with caption sidecars).
      2. Return {'image': Tensor[C,H,W] in [-1,1], 'caption': str}.
    """

    def __init__(
        self,
        root: Optional[str] = None,
        image_size: int = 512,
    ) -> None:
        raise NotImplementedError(
            "AxonometricDataset is not yet implemented. "
            "See data/dataset.py for the expected interface."
        )

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError
