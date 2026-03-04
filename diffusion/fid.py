"""Frechet Inception Distance"""

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from scipy.linalg import sqrtm
from torch.utils.data import DataLoader
from data import DiffusionDataset


transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def calculate_activation_statistics(images, model):
    model.eval()
    with torch.no_grad():
        features = model(images)
    features = features.cpu().numpy()
    mu = np.mean(features, axis = 0)
    sigma = np.cov(features, rowvar = False)
    return mu, sigma


def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(np.dot(sigma1, sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = np.sum(diff**2) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid


if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    inception = models.inception_v3(pretrained=True)
    inception.fc = torch.nn.Identity()

    ds = DiffusionDataset(testing=True)
    dl = DataLoader(ds, batch_size=4, shuffle=False)
    f = []
    for images, _ in dl:
        f.append(inception(images))
    f = torch.cat(f, dim=0)
    mu, sigma = calculate_activation_statistics(f, inception)
    print(mu, sigma)
    print(calculate_fid(mu, sigma, mu, sigma))
