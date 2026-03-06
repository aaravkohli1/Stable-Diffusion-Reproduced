<h1 align="center">UTMIST / Stable Diffusion</h1>


<p align="center">
    <a href="#"><strong>Pretrained Weights</strong></a> · 
    <a href="https://aaravkohli1.github.io/Stable-Diffusion-Reproduced/"><strong>Project Website</strong></a> · 
    <a href="#"><strong>Technical Report</strong></a> 
</p> 

<p align="center"> 
    <img src="https://img.shields.io/badge/Framework-PyTorch%202.1%2B-red" /> 
    <img src="https://img.shields.io/badge/Model-Diffusion-blue" /> 
    <img src="https://img.shields.io/badge/License-MIT-yellow" /> 
    <img src="https://img.shields.io/badge/Status-Active%20Project-brightgreen" />
</p>

## Quick Links

- [Installation](#installation)
- [File Structure](#file-structure)
- [Training](#training)
- [Model Architecture](#model-architecture)
- [Ablations](#ablations)
- [Fine Tuning Experiments](#fine-tuning-experiments)
- [Dataset](#data)

## Installation

```bash
git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced
cd Stable-Diffusion-Reproduced
pip install -r requirements.txt
```

## File Structure


```
Stable-Diffusion-Reproduced/                                                     
  ├── .gitignore                                                                                                                                      
  ├── LICENSE                                                                                                                                         
  ├── README.md                                                                                                                                       
  ├── requirements.txt                                                                                                                                
  │                                                                                                                                                   
  ├── data/                                                                                                               
  │   ├── __init__.py
  │   ├── dataset.py
  │   └── preprocessing.py
  │
  ├── diffusion/
  │   ├── __init__.py
  │   ├── diffuser.py
  │   ├── noise_schedules.py
  │   └── sampling.py
  │
  ├── docs/
  │   ├── index.html
  │   ├── license.html
  │   ├── research.html
  │   └── style.css
  │
  ├── models/
  │   ├── clip/
  │   │   ├── __init__.py
  │   │   ├── clip_text_model.py
  │   │   ├── hf_clip_text.py
  │   │   ├── text_encoder.py
  │   │   ├── tokenizer.py
  │   │   └── transformer.py
  │   ├── unet/
  │   │   └── __init__.py
  │   └── vae/
  │       ├── __init__.py
  │       └── VAE.ipynb
  │
  └── tests/
      ├── __init__.py
      └── test_clip_smoke.py
```

## Training

Training this model is done in several parts. To start, the variational autoencoder (VAE) must be trained as the U-Net operates within its latent space. The U-Net is then trained after, using the VAEs latents on the forward pass.

Note that for each U-Net training iteration, the VAE must compute the latent, and CLIP must compute the text embedding. Thus, one method for reducing memory usage and speeding up training is to precompute these values.

Scripts for precomputing key values, training the VAE, and training the U-Net can be run as follows.

> IMPORTANT NOTE: The University of Toronto Machine Intelligence Student Team is sponsored by Tenstorrent, thus inference scripts for the VAE and U-Net are written for these GPUs specifically.

### Precompute Latents

```bash
git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced
cd Stable-Diffusion-Reproduced
pip install -r requirements.txt
python runs/compute_latents.py
```

### Train VAE

```bash
git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced
cd Stable-Diffusion-Reproduced
pip install -r requirements.txt
python runs/train_vae.py
```
### Train U-Net

```bash
git clone https://github.com/aaravkohli1/Stable-Diffusion-Reproduced
cd Stable-Diffusion-Reproduced
pip install -r requirements.txt
python runs/train_unet.py
```



## Model Architecture

## Ablations

## Fine Tuning Experiments



## Dataset

We use the **LAION aesthetics dataset** for training the individual models, and a **CIFAR-10 Wrapper** for testing inference and training during development. The dataset may be toggled via the `testing` parameter within the `DiffusionDataset` abstraction. For this project, we fine tune on axonometric data specifically. Thus an `FTDataset` abstraction for loading these images is also provided. This is shown below.

### Full Dataset

```python
from data import DiffusionDataset
from torch.utils.data import DataLoader

dataset = DiffusionDataset(testing=False)
loader = DataLoader(dataset, batch_size=4)

...
```

### Testing Data

```python
from data import DiffusionDataset
from torch.utils.data import DataLoader

dataset = DiffusionDataset(testing=True, num_test_samples=10)
loader = DataLoader(dataset, batch_size=4)

...
```

### Fine-Tuning Data
```python
from data import FTDataset
from torch.utils.data import DataLoader

dataset = FTDataset(num_test_samples=10)
loader = DataLoader(dataset, batch_size=4)

...
```



