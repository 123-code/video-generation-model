"""Grain-based data pipeline for loading pre-computed VAE latents (.npy files)."""

import os
import glob
import numpy as np
import grain.python as grain


class LatentDataSource(grain.RandomAccessDataSource):
    """Random-access data source that loads .npy VAE latent files.

    Expects directory structure:
        latent_dir/
        ├── class_0/
        │   ├── video_0.npy
        │   └── ...
        ├── class_1/
        │   └── ...
        └── ...

    Or flat directory for unconditional training.
    """

    def __init__(self, latent_dir: str):
        self.samples = []  # List of (path, label) tuples

        # Look for class subdirectories
        classes = sorted([
            d for d in os.listdir(latent_dir)
            if os.path.isdir(os.path.join(latent_dir, d)) and not d.startswith(".")
        ])

        if not classes:
            # Flat directory (unconditional)
            print("No class folders found. Loading flat directory.")
            files = sorted(glob.glob(os.path.join(latent_dir, "*.npy")))
            self.samples = [(f, 0) for f in files]
            self.num_classes = 0
        else:
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
            print(f"Found classes: {class_to_idx}")
            for cls_name in classes:
                cls_dir = os.path.join(latent_dir, cls_name)
                files = sorted(glob.glob(os.path.join(cls_dir, "*.npy")))
                for f in files:
                    self.samples.append((f, class_to_idx[cls_name]))
            self.num_classes = len(classes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        latent = np.load(path).astype(np.float32)
        return {"latent": latent, "label": np.int32(label)}


def create_dataloader(latent_dir: str, batch_size: int, seed: int = 42):
    """Create a Grain DataLoader for latent files (info only, single-epoch).

    Use create_epoch_dataloader() for each training epoch.

    Args:
        latent_dir: Path to directory containing .npy latent files.
        batch_size: Global batch size.
        seed: Random seed for shuffling.

    Returns:
        dataloader: Iterable that yields batches (dict with 'latent' and 'label').
        num_samples: Total number of samples.
        num_classes: Number of classes found.
    """
    source = LatentDataSource(latent_dir)

    sampler = grain.IndexSampler(
        num_records=len(source),
        num_epochs=1,
        shard_options=grain.ShardByJaxProcess(),
        shuffle=True,
        seed=seed,
    )

    dataloader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        operations=[grain.Batch(batch_size=batch_size, drop_remainder=True)],
        worker_count=4,
    )

    return dataloader, len(source), source.num_classes


def create_epoch_dataloader(latent_dir: str, batch_size: int, seed: int = 42):
    """Create a fresh Grain DataLoader for one epoch of training.

    Should be called at the start of each epoch to get a fresh shuffled iterator.

    Args:
        latent_dir: Path to directory containing .npy latent files.
        batch_size: Global batch size.
        seed: Random seed (vary per epoch for different shuffling).

    Returns:
        dataloader: Iterable that yields batches (dict with 'latent' and 'label').
        num_samples: Total number of samples.
        num_classes: Number of classes found.
    """
    return create_dataloader(latent_dir, batch_size, seed)
