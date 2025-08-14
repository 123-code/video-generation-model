import os
import torch
from torch.utils.data import Dataset


class LatentDataset(Dataset):
    def __init__(self, latent_dir: str):

        self.latent_dir = latent_dir
        self.latent_files = sorted([
            os.path.join(latent_dir, f) for f in os.listdir(latent_dir) if f.endswith('.pt')
        ])
        if len(self.latent_files) == 0:
            raise ValueError(f"No .pt files found in {latent_dir}")

        self.index_map = []  
        for file_idx, fpath in enumerate(self.latent_files):
            tensor = torch.load(fpath, map_location='cpu')
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"Latent file does not contain a tensor: {fpath}")
            if tensor.ndim == 5:
                inner = int(tensor.shape[0])
                for inner_idx in range(inner):
                    self.index_map.append((file_idx, inner_idx))
            elif tensor.ndim == 4:
                self.index_map.append((file_idx, 0))
            else:
                raise ValueError(f"Unsupported latent tensor dims {tensor.ndim} in {fpath}")

    def __len__(self) -> int:
        return len(self.index_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx, inner_idx = self.index_map[idx]
        latent_path = self.latent_files[file_idx]
        latent = torch.load(latent_path, map_location='cpu')

        if latent.ndim == 5:
            latent = latent[inner_idx]
        elif latent.ndim != 4:
            raise ValueError(f"Expected latent dims 4 or 5, got shape {tuple(latent.shape)} in {latent_path}")


        if latent.ndim != 4:
            raise ValueError(f"Expected latent with 4 dims [C,F,H,W], got shape {tuple(latent.shape)} in {latent_path}")
        return latent.float()
