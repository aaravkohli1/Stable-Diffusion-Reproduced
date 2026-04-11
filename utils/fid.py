"""Frechet Inception Distance"""
import torch
from torchvision.transforms import v2
import numpy as np
from tqdm import tqdm
from scipy import linalg


def get_activations(dataset, model):
    preprocess = v2.Compose([
        v2.Resize(299),
        v2.CenterCrop(299),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dl = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    out = None
    for batch in tqdm(dl):
        images = preprocess(batch['image'].to('cpu'))
        with torch.no_grad():
            if out is None:
                out = model(images)
            else:
                out = torch.cat((out, model(images)), dim=0)

    return out.numpy()


def calculate_stats(dataset, model):
    act = get_activations(dataset, model)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_fid(m1, s1, m2, s2):
    # from https://github.com/Hanhpt23/FID-pytorch/blob/master/src/pytorch_fid/fid_score.py
    eps = 1e-6
    diff = m1 - m2
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
                  "fid calculation produces singular product; "
                  "adding %s to diagonal of cov estimates"
              ) % eps
        print(msg)
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * tr_covmean


def fid_from_datasets(dataset1, dataset2, model):
    mu1, sigma1 = calculate_stats(dataset1, model)
    m2, sigma2 = calculate_stats(dataset2, model)
    return calculate_fid(mu1, sigma1, m2, sigma2)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from data import DiffusionDataset

    # Initialize inception v3 model from pytorch
    Iv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
    Iv3.fc = torch.nn.Identity()
    Iv3.eval()

    # Initialize datasets from paths
    # REAL IMAGES (Ideally over 10k images)
    real_ds = DiffusionDataset(testing=True)

    # GENERATED IMAGES
    gen_ds = DiffusionDataset(testing=True)

    fid = fid_from_datasets(real_ds, gen_ds, Iv3)
    print(fid)
