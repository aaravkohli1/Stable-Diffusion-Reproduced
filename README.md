# Stable Diffusion - Reproduced, Ablated, and Fine-tuned

description to come...

## File Structure

Feel free to add folders and files wherever you need for your implementations. This is just a starting point. Individual files don't exist, this is just folder structure.

## Testing Data

The actual dataset is large, so I have provided a CIFAR 10 wrapper for testing.

```python
from data.dataset import DiffusionDataset
from torch.utils.data import DataLoader

dataset = DiffusionDataset(testing=True, num_test_samples=10)
loader = DataLoader(dataset, batch_size=4)

# more pytorch things ...
```

> You might need to change directory paths to access the DiffusionDataset class
