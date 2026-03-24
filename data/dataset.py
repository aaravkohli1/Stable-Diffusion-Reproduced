"""Datasets for training and fine tuning"""

from torch.utils.data import IterableDataset
from datasets import load_dataset
from preprocessing import preprocess_image, preprocess_text

class TrainingData(IterableDataset):

    def __init__(self, shards: int):
        base_url = 'https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-high-quality-captions/resolve/main/data/data-{i:06d}.tar'
        urls = [base_url.format(i=i) for i in range(shards)]
        self.dataset = load_dataset('webdataset', data_files={'train': urls}, split='train', streaming=True)

    def __iter__(self):
        for sample in self.dataset:
            yield {
                'image': preprocess_image(sample.get('jpg'), image_size=512),
                'caption': preprocess_text(sample.get('txt'))
            }
            
# TODO: Implement Fine Tuning Dataset
        
        