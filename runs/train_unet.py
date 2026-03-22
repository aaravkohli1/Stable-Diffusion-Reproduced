import torch
from diffusion import Diffuser, cosine_beta, linear_beta, quadratic_beta
from torch.utils.data import Dataset
from pathlib import Path
from torch.utils.data import DataLoader
from models.unet import UNet
import os

import yaml
import argparse

schedules = {
    "cosine": cosine_beta,
    "linear": linear_beta,
    "quadratic": quadratic_beta
}

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

class CacheDataset(Dataset):
    def __init__(self, path):
        self.cache_dir = Path(path)
        self.latent_paths = sorted(self.cache_dir.glob('*.pt'))

    def __len__(self):
        return len(self.latent_paths)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return torch.load(self.latent_paths[idx])



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    model = UNet(
        in_channels=config['model']["in_channels"],
        out_channels=config['model']['out_channels'],
        channels=config['model']['channels'],
        n_res=config['model']['n_res'],
        channel_mults=config['model']['channel_mults'],
        attention_levels=config['model']['attention_levels'],
        n_heads=config['model']['n_heads'],
        d_cond=512
    ).to(DEVICE)

    dataloader = DataLoader(CacheDataset(config['latents_path']), batch_size=config['training']['batch_size'], shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])

    checkpoints_dir = os.path.join(os.path.dirname(__file__), 'unet_checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    env = Diffuser(timesteps=config['training']['diffusion_steps'], schedule=schedules[config['training']['beta_schedule']])
    batch_tot = 0

    for epoch in range(config['training']['epochs']):
        for step, batch in enumerate(dataloader):
            batch_tot += 1
            optimizer.zero_grad()

            t = torch.randint(0, config['training']['diffusion_steps'], (config['training']['batch_size'],), device=DEVICE).long()

            loss = env.compute_loss(model, batch['latents'], t, conds=batch['conds'])
            loss.backward()
            optimizer.step()

            if batch_tot % config['training']['log_interval'] == 0:
                print(f"Batch {batch_tot} / {len(dataloader)} with loss {loss.item()}")
            if batch_tot % config['training']['save_interval'] == 0:
                path = os.path.join(checkpoints_dir, f"unet_{batch_tot}.pt")
                torch.save(model.state_dict(), path)
                print(f"Checkpoint saved: {path}")

    final_path = os.path.join(checkpoints_dir, "unet_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")
