import os
import torch
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        """
        A dataset for loading pre-computed latent tensors.
        
        Args:
            latent_dir (str): The directory containing the saved .pt latent files.
        """
        self.latent_dir = latent_dir
        self.latent_files = sorted([os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')])
        if len(self.latent_files) == 0:
            raise ValueError(f"No .pt files found in {latent_dir}")

    def __len__(self):
        return len(self.latent_files)

    def __getitem__(self, idx):
        """
        Loads a single latent tensor.
        """
        latent_path = self.latent_files[idx]
        # The loaded tensor will be on the CPU, the DataLoader will move it to the GPU
        latent = torch.load(latent_path)
        return latent