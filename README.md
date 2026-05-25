<h1 align="center">UTMIST / Stable Diffusion</h1>

<p align="center">
    <a href="https://aaravkohli1.github.io/Stable-Diffusion-Reproduced/"><strong>Project Website</strong></a> ·
    <a href="https://aaravkohli1.github.io/Stable-Diffusion-Reproduced/research.html"><strong>Implementation</strong></a> ·
    <a href="#inference">Inference</a> ·
    <a href="#training">Training</a>
</p>

<p align="center">
    <img src="https://img.shields.io/badge/Framework-PyTorch%202.11-red" />
    <img src="https://img.shields.io/badge/Model-Latent%20Diffusion-blue" />
    <img src="https://img.shields.io/badge/License-MIT-yellow" />
</p>

A from-scratch reimplementation of Stable Diffusion by the University of Toronto Machine Intelligence Student Team. Includes a working VAE / U-Net / CLIP-text stack, five samplers (DDPM, DDIM, Euler-Ancestral, Heun, DPM-Solver++ 2M), four β schedules, classifier-free guidance with the rescale fix, and a weights-converter that loads official SD 1.4 checkpoints into our modules.

## Quick Links

- [Installation](#installation)
- [Inference](#inference)
- [Training](#training)
- [Datasets](#datasets)
- [File Structure](#file-structure)

## Installation

```bash
git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced
cd Stable-Diffusion-Reproduced
pip install -r requirements.txt
# for loading pretrained SD 1.4 weights:
pip install diffusers transformers accelerate safetensors
```

A CUDA-enabled PyTorch build is strongly recommended.

## Inference

Generate a 512×512 image using the official SD 1.4 weights loaded through our modules:

```bash
python -m models.generate \
    --prompt "a corgi sitting in a meadow of wildflowers" \
    --from-pretrained CompVis/stable-diffusion-v1-4 \
    --sampler dpm_pp_2m --steps 25 --guidance-scale 7.5 --karras \
    --output corgi.png
```

The script supports five samplers (`ddpm`, `ddim`, `euler_a`, `heun`, `dpm_pp_2m`), CFG-rescale (`--cfg-rescale 0.7`), the Karras σ schedule (`--karras`), and four β schedules (`linear`, `scaled_linear`, `cosine`, `quadratic`). Pass `--vae-ckpt` / `--unet-ckpt` to load your own trained weights instead.

To batch-render the showcase grid that powers the project website:

```bash
python -m runs.batch_generate --out docs/assets
```

## Training

Training proceeds in three stages: cache latents and text embeddings, train the VAE, then train the U-Net on cached latents.

### 1. Precompute latents and text embeddings

```bash
python -m runs.compute_latents \
    --out runs/latent_cache --num-samples 5000 --image-size 256 \
    --vae-ckpt runs/checkpoints/vae_final.pt
```

Use `--testing` to run against the CIFAR-10 wrapper for a fast end-to-end smoke check (no network shards required).

### 2. Train the VAE

```bash
python -m runs.train_vae
```

Trains the VAE on `DiffusionDataset`, saving checkpoints into `runs/checkpoints/`. Hyperparameters live at the top of the script.

### 3. Train the U-Net

```bash
python -m runs.train_unet configs/unet_base.yaml
```

`configs/unet_base.yaml` selects the model dims, β schedule (`linear` / `cosine` / `quadratic`), batch size, learning rate, and logging cadence. The dataset reads cached `(latents, conds)` pairs from step 1.

## Datasets

`DiffusionDataset` streams a LAION-style synthetic high-quality captions shard set, or a CIFAR-10 wrapper for smoke tests when `testing=True`. `FTDataset` reads a local image folder.

### Full Dataset

```python
from data import DiffusionDataset
from torch.utils.data import DataLoader

dataset = DiffusionDataset(testing=False, image_size=512)
loader = DataLoader(dataset, batch_size=4)
```

### Testing Data

```python
from data import DiffusionDataset
from torch.utils.data import DataLoader

dataset = DiffusionDataset(testing=True, num_test_samples=100, image_size=256)
loader = DataLoader(dataset, batch_size=4)
```

### Local Image Folder

```python
from data import FTDataset
from torch.utils.data import DataLoader

dataset = FTDataset(root="my_images", image_size=512)
loader = DataLoader(dataset, batch_size=4)
```

## File Structure

```
Stable-Diffusion-Reproduced/
├── data/                  # DiffusionDataset + FTDataset, preprocessing
├── diffusion/             # forward process, β schedules, samplers
├── docs/                  # project website (GitHub Pages)
├── models/
│   ├── clip/              # tokenizer + transformer text encoder
│   ├── unet.py            # U-Net backbone
│   ├── unet_attention.py  # cross/self-attention blocks
│   ├── vae.py             # VAE encoder/decoder
│   ├── convert_weights.py # diffusers → our state_dict mapping
│   └── generate.py        # text-to-image inference entrypoint
├── runs/
│   ├── compute_latents.py # cache VAE + CLIP outputs
│   ├── train_vae.py
│   ├── train_unet.py
│   └── batch_generate.py  # showcase grid renderer
├── utils/                 # FID, attention-map visualization
├── tests/                 # CLIP smoke, CFG, flash-attn, attn-visual
├── LICENSE
├── README.md
└── requirements.txt
```
