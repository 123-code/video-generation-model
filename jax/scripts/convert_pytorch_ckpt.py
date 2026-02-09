"""Convert a PyTorch VideoDiTV2 checkpoint to JAX-compatible format.

Handles the key differences:
- PyTorch Linear weights are (out_features, in_features)
- Flax/JAX Linear weights are (in_features, out_features) -> transpose needed
- Embedding weights have same layout in both frameworks
- RMSNorm weight is 1D in both frameworks

Usage:
    python scripts/convert_pytorch_ckpt.py \
        --pytorch_ckpt path/to/checkpoint.pt \
        --output_dir jax_checkpoint \
        --num_classes 5
"""

import os
import argparse
import numpy as np


def convert_pytorch_to_jax(pytorch_ckpt_path: str, output_dir: str,
                           dim=1024, depth=16, heads=16, dim_head=64,
                           num_classes=5):
    """Convert PyTorch checkpoint to JAX-compatible numpy arrays.

    Args:
        pytorch_ckpt_path: Path to .pt checkpoint file.
        output_dir: Directory to save converted weights.
        dim: Model dimension.
        depth: Number of transformer blocks.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        num_classes: Number of classes.
    """
    import torch

    print(f"Loading PyTorch checkpoint: {pytorch_ckpt_path}")
    ckpt = torch.load(pytorch_ckpt_path, map_location="cpu")

    # Handle different checkpoint formats
    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "shadow" in ckpt:
        # EMA checkpoint
        state_dict = ckpt["shadow"]
    else:
        state_dict = ckpt

    jax_weights = {}

    for key, tensor in state_dict.items():
        np_array = tensor.numpy()

        # Determine if this is a linear weight that needs transposing
        # Linear weights in PyTorch: (out_features, in_features)
        # Linear weights in JAX/Flax: (in_features, out_features)
        needs_transpose = False

        if "weight" in key and np_array.ndim == 2:
            # This is a linear layer weight -> transpose
            needs_transpose = True
        elif "embedding" in key.lower() or "embed" in key.lower():
            # Embedding weights: same layout (num_embeddings, dim)
            needs_transpose = False

        if needs_transpose:
            np_array = np_array.T

        # Map PyTorch key to JAX key structure
        jax_key = _map_key(key)
        jax_weights[jax_key] = np_array
        print(f"  {key} ({tensor.shape}) -> {jax_key} ({np_array.shape})"
              f" {'[transposed]' if needs_transpose else ''}")

    # Save as numpy arrays
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "converted_weights.npz")
    np.savez(output_path, **jax_weights)
    print(f"\nSaved {len(jax_weights)} tensors to {output_path}")

    # Also save metadata
    meta_path = os.path.join(output_dir, "metadata.npz")
    np.savez(meta_path, dim=dim, depth=depth, heads=heads, dim_head=dim_head,
             num_classes=num_classes)
    print(f"Saved metadata to {meta_path}")


def _map_key(pytorch_key: str) -> str:
    """Map a PyTorch state_dict key to a JAX-compatible key.

    The exact mapping depends on how the Flax NNX model names its parameters.
    This provides a reasonable mapping that can be manually adjusted.
    """
    # Replace PyTorch naming conventions with Flax-style
    key = pytorch_key

    # Handle ModuleList indexing: blocks.0. -> blocks/0/
    key = key.replace(".", "/")

    # Common renames
    key = key.replace("/weight", "/kernel")  # Linear weight -> kernel
    # Embedding weight stays as "embedding" in NNX

    return key


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoint to JAX")
    parser.add_argument("--pytorch_ckpt", type=str, required=True,
                        help="Path to PyTorch .pt checkpoint")
    parser.add_argument("--output_dir", type=str, default="jax_checkpoint",
                        help="Output directory for converted weights")
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--num_classes", type=int, default=5)
    args = parser.parse_args()

    convert_pytorch_to_jax(
        args.pytorch_ckpt, args.output_dir,
        dim=args.dim, depth=args.depth, heads=args.heads,
        dim_head=args.dim_head, num_classes=args.num_classes,
    )


if __name__ == "__main__":
    main()
