"""Inference script: generate videos from a trained VideoDiTV2 checkpoint."""

import os
import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import imageio
import orbax.checkpoint as ocp

from configs.default import get_config
from model.video_dit import VideoDiTV2
from diffusion.schedules import create_cosine_schedule, create_linear_schedule
from diffusion.sampling import ddpm_sample, ddim_sample


def load_vae():
    """Load Stable Diffusion VAE for decoding latents to pixels.

    Requires diffusers + torch (only needed for generation, not training).
    """
    import torch
    from diffusers import AutoencoderKL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    return vae, device


def decode_latents_with_vae(vae, z_np, torch_device):
    """Decode JAX latent array through PyTorch VAE.

    Args:
        vae: Stable Diffusion VAE model.
        z_np: Numpy array of latents (B, C, T, H, W).
        torch_device: PyTorch device.

    Returns:
        Numpy array of videos (B, T, H_px, W_px, 3), uint8 [0, 255].
    """
    import torch

    B, C, T, H, W = z_np.shape
    z_torch = torch.from_numpy(z_np).to(torch_device)

    # Unscale latents (reverse VAE scaling)
    z_flat = z_torch.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    z_flat = z_flat / 0.18215

    with torch.no_grad():
        video_flat = vae.decode(z_flat).sample

    video = video_flat.reshape(B, T, 3, H * 8, W * 8).permute(0, 1, 3, 4, 2)
    video = torch.clamp((video + 1.0) / 2.0, 0, 1)
    return (video.cpu().numpy() * 255).astype(np.uint8)


def save_video(frames, path, fps=8):
    """Save a list of frames as an MP4 video."""
    imageio.mimwrite(path, list(frames), fps=fps, codec="libx264", quality=8)


def main():
    parser = argparse.ArgumentParser(description="Generate videos with VideoDiTV2")
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                        help="Orbax checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--use_ddpm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Generate samples for a specific class")
    parser.add_argument("--decode_vae", action="store_true",
                        help="Decode latents to video with SD VAE (requires torch + diffusers)")

    # Model config (must match training)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--num_classes", type=int, default=5)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    config = get_config()

    # --- Model ---
    rngs = nnx.Rngs(args.seed)
    model = VideoDiTV2(
        in_channels=4, T=16, H=32, W=32, patch_size=2,
        dim=args.dim, depth=args.depth, heads=args.heads, dim_head=64,
        dropout=0.0, num_classes=args.num_classes, rngs=rngs,
    )

    # --- Load checkpoint ---
    manager = ocp.CheckpointManager(os.path.abspath(args.checkpoint_dir))
    latest_step = manager.latest_step()
    if latest_step is None:
        print("No checkpoint found!")
        return

    print(f"Loading checkpoint from step {latest_step}...")
    model_params = nnx.state(model, nnx.Param)
    abstract_state = {"ema_params": model_params}
    restored = manager.restore(latest_step, args=ocp.args.StandardRestore(abstract_state))
    nnx.update(model, restored["ema_params"])
    print("Checkpoint loaded (EMA params).")

    # --- Diffusion schedule ---
    schedule = create_cosine_schedule(config.diffusion.timesteps)

    # --- Optional VAE ---
    vae, torch_device = None, None
    if args.decode_vae:
        print("Loading VAE...")
        vae, torch_device = load_vae()

    # --- Generate ---
    rng = jax.random.key(args.seed)
    shape = (1, 4, 16, 32, 32)
    generated = 0

    classes = [args.class_idx] if args.class_idx is not None else list(range(args.num_classes))

    print(f"Generating {args.num_samples} samples per class...")
    for class_idx in classes:
        for sample_i in range(args.num_samples):
            rng, sample_rng = jax.random.split(rng)
            y = jnp.array([class_idx], dtype=jnp.int32)

            if args.use_ddpm:
                z = ddpm_sample(schedule, model, shape, y=y,
                                cfg_scale=args.cfg_scale, rng=sample_rng)
            else:
                z = ddim_sample(schedule, model, shape, y=y,
                                cfg_scale=args.cfg_scale,
                                ddim_steps=args.ddim_steps,
                                rng=sample_rng)

            z_np = np.array(z)

            # Save raw latents
            npy_path = os.path.join(args.output_dir, f"class{class_idx}_sample{sample_i}.npy")
            np.save(npy_path, z_np)

            # Decode and save video if VAE available
            if vae is not None:
                video = decode_latents_with_vae(vae, z_np, torch_device)
                mp4_path = os.path.join(args.output_dir, f"class{class_idx}_sample{sample_i}.mp4")
                save_video(video[0], mp4_path)
                print(f"  Saved: {mp4_path}")
            else:
                print(f"  Saved latent: {npy_path}")

            generated += 1

    print(f"Done! Generated {generated} samples in {args.output_dir}")


if __name__ == "__main__":
    main()
