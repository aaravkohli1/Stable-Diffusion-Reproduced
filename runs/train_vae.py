"""Training script for VAE model"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.vae import VAE
from data.dataset import DiffusionDataset

# Hyper parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 10
KL_WEIGHT = 1e-6
NUM_SAMPLES = 100
IMAGE_SIZE = 256
LOG_EVERY = 10        # print loss every N steps
SAVE_EVERY = 500      # save checkpoint every N steps

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

def train():
    print(f"Using device: {DEVICE}")

    dataset = DiffusionDataset(
        testing=False,
        num_test_samples=NUM_SAMPLES,
        image_size=IMAGE_SIZE
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    model = VAE().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        epoch_steps = 0

        for batch in dataloader:
            images = batch['image'].to(DEVICE)

            decoded, encoded, mean, log_var = model(images)

            recon_loss = F.mse_loss(decoded, images)

            kl_loss = -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

            loss = recon_loss + KL_WEIGHT * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_steps += 1

            if global_step % LOG_EVERY == 0:
                print(
                    f"Epoch {epoch+1}/{EPOCHS} | Step {global_step} | "
                    f"Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | "
                    f"KL: {kl_loss.item():.4f}"
                )

            if global_step % SAVE_EVERY == 0:
                path = os.path.join(ckpt_dir, f"vae_step_{global_step}.pt")
                torch.save(model.state_dict(), path)
                print(f"Checkpoint saved: {path}")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1}/{EPOCHS} complete | Avg Loss: {avg_loss:.4f}")

    final_path = os.path.join(ckpt_dir, "vae_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved: {final_path}")

if __name__ == '__main__':
    train()
