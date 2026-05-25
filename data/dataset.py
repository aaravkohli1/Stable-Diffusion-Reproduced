"""Datasets for training and fine tuning.

Three abstractions:
- ``DiffusionDataset``: streams the LAION-style high-quality captions shards for full
  training, or a tiny CIFAR-10 wrapper for fast smoke tests (``testing=True``).
- ``FTDataset``: loads our local axonometric fine-tuning images.
- ``TrainingData``: legacy alias for ``DiffusionDataset`` kept for backward compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from .preprocessing import preprocess_image, preprocess_text


_LAION_SHARDS = (
    "https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions"
    "/resolve/main/data/data-{i:06d}.tar"
)


class DiffusionDataset(IterableDataset):
    """Training dataset.

    With ``testing=True`` wraps CIFAR-10 (small, fast, no network) so unit tests and
    smoke runs do not require streaming the full shard set.
    """

    def __init__(
        self,
        testing: bool = False,
        num_test_samples: int = 100,
        image_size: int = 512,
        shards: int = 1,
    ) -> None:
        self.testing = testing
        self.image_size = image_size
        self.num_test_samples = num_test_samples
        self._dataset = None  # lazy

        if not testing:
            from datasets import load_dataset
            urls = [_LAION_SHARDS.format(i=i) for i in range(shards)]
            self._dataset = load_dataset(
                "webdataset", data_files={"train": urls}, split="train", streaming=True,
            )

    def _iter_test(self) -> Iterator[dict]:
        from torchvision.datasets import CIFAR10
        cifar = CIFAR10(root="data/cifar10", train=True, download=True)
        # CIFAR-10 class names act as captions; tiny but enough for an end-to-end smoke run.
        for i in range(min(self.num_test_samples, len(cifar))):
            img, label = cifar[i]
            yield {
                "image": preprocess_image(img, image_size=self.image_size),
                "caption": preprocess_text(cifar.classes[label]),
            }

    def __iter__(self) -> Iterator[dict]:
        if self.testing:
            yield from self._iter_test()
            return
        for sample in self._dataset:
            yield {
                "image": preprocess_image(sample.get("jpg"), image_size=self.image_size),
                "caption": preprocess_text(sample.get("txt") or ""),
            }


class FTDataset(Dataset):
    """Fine-tuning dataset over a local image folder (e.g. ``axonometric/``).

    Captions are pulled from a sibling ``.txt`` file when present and otherwise fall back
    to the parent directory name.
    """

    def __init__(
        self,
        root: str = "axonometric",
        image_size: int = 512,
        num_test_samples: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.image_size = image_size
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in exts)
        if num_test_samples is not None:
            paths = paths[:num_test_samples]
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        caption_path = path.with_suffix(".txt")
        caption = caption_path.read_text().strip() if caption_path.exists() else path.parent.name
        return {
            "image": preprocess_image(img, image_size=self.image_size),
            "caption": preprocess_text(caption),
        }


# Backward-compatible alias for earlier code that imported ``TrainingData``.
TrainingData = DiffusionDataset
