"""Main training loop for VideoDiTV2 on JAX/TPU."""

import os
import time
import argparse

import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
from tqdm import tqdm

from configs.default import get_config
from model.video_dit import VideoDiTV2
from diffusion.schedules import create_cosine_schedule, create_linear_schedule
from diffusion.sampling import ddim_sample
from training.train_state import create_optimizer
from training.train_step import train_step
from data.dataset import create_dataloader, create_epoch_dataloader
from checkpointing.checkpoint_manager import create_checkpoint_manager, save_checkpoint, restore_checkpoint
from checkpointing.hf_upload import upload_checkpoint_to_hf


def generate_samples(model, ema_params, schedule, config, epoch, sample_dir, rng):
    """Generate sample videos using EMA parameters for visual inspection."""
    os.makedirs(sample_dir, exist_ok=True)

    # Temporarily swap model params with EMA params
    original_params = nnx.state(model, nnx.Param)
    nnx.update(model, ema_params)

    classes_to_gen = list(range(min(config.model.num_classes, 5)))
    shape = (1, config.model.in_channels, config.model.T,
             config.model.H, config.model.W)

    for class_idx in classes_to_gen:
        rng, sample_rng = jax.random.split(rng)
        y = jnp.array([class_idx], dtype=jnp.int32)

        z = ddim_sample(
            schedule, model, shape,
            y=y, cfg_scale=config.sampling.cfg_scale,
            ddim_steps=config.sampling.ddim_steps,
            eta=config.sampling.eta,
            rng=sample_rng,
        )

        # Save raw latents as .npy for later VAE decoding
        out_path = os.path.join(sample_dir, f"epoch_{epoch:03d}_class_{class_idx}.npy")
        np.save(out_path, np.array(z))
        print(f"  Saved sample: {out_path}")

    # Restore original params
    nnx.update(model, original_params)
    return rng


def main():
    parser = argparse.ArgumentParser(description="Train VideoDiTV2 on JAX/TPU")
    parser.add_argument("--config", type=str, default=None, help="Path to config override")
    parser.add_argument("--latent_dir", type=str, default=None, help="Override latent directory")
    parser.add_argument("--fresh_start", action="store_true", help="Ignore existing checkpoints")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of classes")
    args = parser.parse_args()

    config = get_config()

    # Apply overrides
    if args.latent_dir:
        config.data.latent_dir = args.latent_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.epochs:
        config.training.epochs = args.epochs
    if args.num_classes:
        config.model.num_classes = args.num_classes

    # --- Device setup ---
    devices = jax.devices()
    num_devices = len(devices)
    print(f"JAX devices: {num_devices} x {devices[0].platform}")
    print(f"Global batch size: {config.training.batch_size}")
    assert config.training.batch_size % num_devices == 0, \
        f"Batch size {config.training.batch_size} must be divisible by {num_devices} devices"

    # Data parallelism mesh
    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))

    # --- Data (get sample count) ---
    print(f"Loading latents from: {config.data.latent_dir}")
    _, num_samples, num_classes_found = create_dataloader(
        config.data.latent_dir,
        batch_size=config.training.batch_size,
    )
    print(f"Found {num_samples} samples, {num_classes_found} classes")

    steps_per_epoch = num_samples // config.training.batch_size
    total_steps = steps_per_epoch * config.training.epochs
    print(f"Steps per epoch: {steps_per_epoch}, Total steps: {total_steps}")

    # --- Model ---
    rngs = nnx.Rngs(config.training.seed)
    model = VideoDiTV2(
        in_channels=config.model.in_channels,
        T=config.model.T,
        H=config.model.H,
        W=config.model.W,
        patch_size=config.model.patch_size,
        dim=config.model.dim,
        depth=config.model.depth,
        heads=config.model.heads,
        dim_head=config.model.dim_head,
        mlp_mult=config.model.mlp_mult,
        dropout=config.model.dropout,
        num_classes=config.model.num_classes,
        rngs=rngs,
    )

    # Count parameters
    param_count = sum(p.size for p in jax.tree.leaves(nnx.state(model, nnx.Param)))
    print(f"Model parameters: {param_count:,}")

    # --- Optimizer (nnx.Optimizer wraps optax) ---
    optimizer = create_optimizer(
        model,
        lr=config.training.lr,
        min_lr=config.training.min_lr,
        total_steps=total_steps,
        weight_decay=config.training.weight_decay,
        grad_clip_norm=config.training.grad_clip_norm,
    )

    # --- EMA ---
    ema_params = jax.tree.map(lambda p: p.copy(), nnx.state(model, nnx.Param))

    # --- Diffusion schedule ---
    if config.diffusion.beta_schedule == "cosine":
        schedule = create_cosine_schedule(config.diffusion.timesteps)
    else:
        schedule = create_linear_schedule(config.diffusion.timesteps)

    # Pre-cast schedule arrays to float32 for TPU compatibility
    sqrt_alphas_cumprod = schedule.sqrt_alphas_cumprod.astype(jnp.float32)
    sqrt_one_minus_alphas_cumprod = schedule.sqrt_one_minus_alphas_cumprod.astype(jnp.float32)

    # --- Checkpointing ---
    ckpt_manager = create_checkpoint_manager(
        config.checkpoint.checkpoint_dir,
        max_to_keep=config.checkpoint.max_to_keep,
    )

    start_epoch = 0
    global_step = 0
    rng = jax.random.key(config.training.seed)

    # --- Resume from checkpoint ---
    if not args.fresh_start:
        restored = restore_checkpoint(ckpt_manager, model, optimizer, ema_params)
        if restored is not None:
            model, optimizer, ema_params, start_epoch, global_step, rng = restored
            print(f"Resumed from epoch {start_epoch}, step {global_step}")
        else:
            print("No checkpoint found. Starting fresh.")

    # --- Training loop ---
    print(f"\nStarting training from epoch {start_epoch}...")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  LR: {config.training.lr}")
    print(f"  Weight decay: {config.training.weight_decay}")
    print(f"  EMA decay: {config.ema.decay}")
    print(f"  CFG dropout: {config.training.cfg_dropout_prob}")

    for epoch in range(start_epoch, config.training.epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t_start = time.time()

        # Create a fresh dataloader each epoch (Grain sampler is single-epoch)
        dataloader, _, _ = create_epoch_dataloader(
            config.data.latent_dir,
            batch_size=config.training.batch_size,
            seed=config.training.seed + epoch,
        )

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.training.epochs}",
                          total=steps_per_epoch):
            latents, labels = batch["latent"], batch["label"]

            # Convert numpy arrays to jax arrays and shard across devices
            latents = jax.device_put(jnp.array(latents), data_sharding)
            labels = jax.device_put(jnp.array(labels), data_sharding)

            loss, ema_params, rng = train_step(
                model, optimizer, ema_params,
                sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod,
                schedule.timesteps,
                latents, labels, rng, global_step,
                config.training.cfg_dropout_prob,
                config.ema.decay, config.ema.warmup_steps,
            )

            global_step += 1
            epoch_loss += float(loss)
            epoch_steps += 1

            # Step checkpoint upload
            if global_step % config.checkpoint.save_every_steps == 0:
                save_checkpoint(
                    ckpt_manager, model, optimizer, ema_params,
                    epoch, global_step, rng,
                )
                try:
                    upload_checkpoint_to_hf(
                        config.checkpoint.checkpoint_dir,
                        config.checkpoint.hf_repo_id,
                        global_step, epoch,
                    )
                except Exception as e:
                    print(f"  HF upload failed: {e}")

        # Epoch stats
        elapsed = time.time() - t_start
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, time={elapsed:.1f}s, "
              f"steps/s={epoch_steps / elapsed:.1f}")

        # Full checkpoint to HF
        if (epoch + 1) % config.checkpoint.save_every_epochs == 0:
            save_checkpoint(
                ckpt_manager, model, optimizer, ema_params,
                epoch, global_step, rng,
            )
            try:
                upload_checkpoint_to_hf(
                    config.checkpoint.checkpoint_dir,
                    config.checkpoint.hf_repo_id,
                    global_step, epoch + 1,
                )
                print(f"  Epoch {epoch + 1} checkpoint uploaded to HuggingFace")
            except Exception as e:
                print(f"  HF upload failed: {e}")

        # Generate samples
        if (epoch + 1) % config.checkpoint.sample_every_epochs == 0:
            print(f"  Generating samples at epoch {epoch + 1}...")
            rng = generate_samples(
                model, ema_params, schedule, config,
                epoch + 1, config.checkpoint.sample_dir, rng,
            )

    print("Training complete.")


if __name__ == "__main__":
    main()
