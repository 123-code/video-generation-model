"""One-time conversion: .pt latent files -> .npy for JAX training.

Eliminates PyTorch dependency from the JAX training pipeline.

Usage:
    python data/convert_pt_to_npy.py --input_dir latents --output_dir latents_npy
"""

import os
import glob
import argparse
import numpy as np


def convert_pt_to_npy(input_dir: str, output_dir: str):
    """Convert all .pt files to .npy, preserving directory structure.

    Args:
        input_dir: Source directory with .pt files (possibly in class subdirs).
        output_dir: Destination directory for .npy files.
    """
    # Lazy import torch -- only needed for this conversion script
    import torch

    os.makedirs(output_dir, exist_ok=True)

    # Find all .pt files
    pt_files = glob.glob(os.path.join(input_dir, "**", "*.pt"), recursive=True)

    if not pt_files:
        print(f"No .pt files found in {input_dir}")
        return

    print(f"Converting {len(pt_files)} files from {input_dir} -> {output_dir}")

    for i, pt_path in enumerate(pt_files):
        # Compute relative path to preserve class subdirectory structure
        rel_path = os.path.relpath(pt_path, input_dir)
        npy_path = os.path.join(output_dir, rel_path.replace(".pt", ".npy"))

        # Create subdirectory if needed
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)

        # Load .pt and save as .npy
        tensor = torch.load(pt_path, map_location="cpu")
        if hasattr(tensor, "numpy"):
            array = tensor.numpy()
        else:
            array = np.array(tensor)

        np.save(npy_path, array.astype(np.float32))

        if (i + 1) % 100 == 0 or i == 0:
            print(f"  [{i + 1}/{len(pt_files)}] {rel_path} -> shape {array.shape}")

    print(f"Done. Converted {len(pt_files)} files.")


def main():
    parser = argparse.ArgumentParser(description="Convert .pt latents to .npy")
    parser.add_argument("--input_dir", type=str, default="latents",
                        help="Source directory with .pt files")
    parser.add_argument("--output_dir", type=str, default="latents_npy",
                        help="Destination directory for .npy files")
    args = parser.parse_args()
    convert_pt_to_npy(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
