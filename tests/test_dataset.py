"""
Test the Dataset Utilities (Run with PyTest)
Run this file directly to visualize the data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from data import DiffusionDataset


# Helper

def make_cifar(n=8):
    return DiffusionDataset(testing=True, num_test_samples=n, image_size=64)


# CIFAR 10

def test_cifar_length():
    ds = make_cifar(16)
    assert len(ds) == 16


def test_cifar_item_keys():
    ds = make_cifar()
    item = ds[0]
    assert set(item.keys()) == {'image', 'caption', 'image_path'}


def test_cifar_image_shape():
    ds = make_cifar()
    image = ds[0]['image']
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)


def test_cifar_image_range():
    """preprocess_image normalises to [-1, 1]"""
    ds = make_cifar(32)
    for i in range(len(ds)):
        img = ds[i]['image']
        assert img.min() >= -1.1 and img.max() <= 1.1


def test_cifar_caption_is_string():
    ds = make_cifar()
    caption = ds[0]['caption']
    assert isinstance(caption, str) and len(caption) > 0


def test_cifar_caption_references_class():
    class_names = {
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck',
    }
    ds = make_cifar(50)
    for i in range(len(ds)):
        caption = ds[i]['caption']
        assert any(cls in caption for cls in class_names), \
            f"Caption '{caption}' doesn't reference a known CIFAR class"


def test_cifar_dataloader():
    ds = make_cifar(8)
    loader = DataLoader(ds, batch_size=4)
    batch = next(iter(loader))
    assert batch['image'].shape == (4, 3, 64, 64)
    assert len(batch['caption']) == 4


def test_cifar_image_path_format():
    ds = make_cifar(4)
    for i in range(len(ds)):
        assert ds[i]['image_path'].startswith('cifar10_')


# LAION 

def make_laion(n=5):
    return DiffusionDataset(testing=False, num_test_samples=n, image_size=64)


def test_laion_length():
    ds = make_laion(5)
    assert len(ds) == 5


def test_laion_item_keys():
    ds = make_laion(3)
    item = ds[0]
    assert set(item.keys()) == {'image', 'caption', 'image_path'}


def test_laion_image_shape():
    ds = make_laion(3)
    image = ds[0]['image']
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)


def test_laion_image_range():
    ds = make_laion(5)
    for i in range(len(ds)):
        img = ds[i]['image']
        assert img.min() >= -1.1 and img.max() <= 1.1


def test_laion_caption_is_string():
    ds = make_laion(3)
    for i in range(len(ds)):
        caption = ds[i]['caption']
        assert isinstance(caption, str) and len(caption) > 0


def test_laion_dataloader():
    ds = make_laion(4)
    loader = DataLoader(ds, batch_size=2)
    batch = next(iter(loader))
    assert batch['image'].shape == (2, 3, 64, 64)
    assert len(batch['caption']) == 2


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ds = DiffusionDataset(testing=False, num_test_samples=5, image_size=256)

    fig, axes = plt.subplots(1, len(ds), figsize=(4 * len(ds), 4))
    for i, ax in enumerate(axes):
        item = ds[i]
        img = (item['image'].permute(1, 2, 0) + 1) / 2
        ax.imshow(img.clamp(0, 1))
        ax.set_title(item['caption'][:40], fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
