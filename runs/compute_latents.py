from models.vae import Encoder
from models.clip import CLIPTextEncoder
from data import DiffusionDataset
import torch
import os

WEIGHTS_PATH = ''
SAVE_PATH = './latent_cache'

os.makedirs(SAVE_PATH, exist_ok=True)

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

ve = Encoder().to(DEVICE)
if WEIGHTS_PATH:
    ve.load_state_dict(torch.load(WEIGHTS_PATH, weights_only=True))
ve.eval()

clip = CLIPTextEncoder.from_pretrained_hf("openai/clip-vit-base-patch32", max_length=64).to(DEVICE)
clip.eval()

batch_count = 1
batch_size = 4

dd = DiffusionDataset(num_test_samples=batch_size*batch_count, image_size=512)

for i in range(batch_count):
    imgs = torch.stack([dd[_ + i * batch_size]['image'] for _ in range(batch_size)])
    with torch.no_grad():
        conds = clip.encode([dd[_ + i * batch_size]['caption'] for _ in range(batch_size)])
        latents, m, v = ve(imgs.to(DEVICE))

    for d in range(batch_size):
        torch.save({"conds": conds[d, :, :], "latents": latents[d, :, :, :]}, f"{SAVE_PATH}/{i * 32 + d}.pt")
    print(f"Batch {batch_count} done.")
