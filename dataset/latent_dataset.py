import torch
from torch.utils.data import Dataset
import os
import glob

class LatentDataset(Dataset):
    def __init__(self, latent_dir):
        self.latent_dir = latent_dir
        self.samples = []
        
        # 1. Look for class folders first
        # Sort to ensure index 0=abstract, 1=clouds, etc.
        self.classes = sorted([d for d in os.listdir(latent_dir) if os.path.isdir(os.path.join(latent_dir, d)) and not d.startswith('.')])
        
        if not self.classes:
            # Fallback: Flat directory (Unconditional)
            print("No class folders found. Loading flat directory.")
            self.samples = [(f, 0) for f in glob.glob(os.path.join(latent_dir, "*.pt"))]
        else:
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
            print(f"Found classes: {self.class_to_idx}")

            # Gather files from subfolders
            for class_name in self.classes:
                class_dir = os.path.join(latent_dir, class_name)
                files = glob.glob(os.path.join(class_dir, "*.pt"))
                for f in files:
                    self.samples.append((f, self.class_to_idx[class_name]))
                
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # Load latent
        latent = torch.load(path, map_location='cpu') # Load to CPU first to save GPU VRAM during dataloading
        return latent.float(), label