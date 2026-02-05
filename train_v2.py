import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
import imageio
from datetime import datetime
from huggingface_hub import HfApi, hf_hub_download, list_repo_files, upload_file

from model import VideoDiTV2
from diffusion import GaussianDiffusion, EMA
from dataset import LatentDataset

# HuggingFace configuration
REPO_ID = "Jnaranjo/video-dit-spot"  # Repo to save checkpoints
CHECKPOINT_REPO_ID = "Jnaranjo/video-generation-epoch90-checkpoint"  # Repo to resume from
SAVE_EVERY_STEPS = 500
HF_CHECKPOINT_EVERY_EPOCHS = 5  # Save full checkpoint to HF every N epochs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# H100 optimizations
if torch.cuda.is_available():
    # Enable TF32 for faster matmuls on Ampere+ GPUs (H100, A100)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Enable cudnn autotuning for optimal convolution algorithms
    torch.backends.cudnn.benchmark = True

def get_latest_checkpoint(prefer_checkpoint_repo=True):
    """
    Get latest checkpoint, first checking CHECKPOINT_REPO_ID, then REPO_ID.
    """
    repos_to_check = [CHECKPOINT_REPO_ID, REPO_ID] if prefer_checkpoint_repo else [REPO_ID]

    for repo in repos_to_check:
        try:
            files = list_repo_files(repo)
            # Look for epoch checkpoints first (more reliable), then step checkpoints
            epoch_ckpts = sorted([f for f in files if f.endswith(".pt") and "epoch" in f.lower()])
            step_ckpts = sorted([f for f in files if f.endswith(".pt") and "checkpoint-" in f])

            ckpts = epoch_ckpts if epoch_ckpts else step_ckpts
            if not ckpts:
                continue

            latest = ckpts[-1]
            print(f"Found checkpoint: {latest} in {repo}")
            local_path = hf_hub_download(repo_id=repo, filename=latest)
            print(f"Resuming from: {latest}")
            return torch.load(local_path, map_location=device)
        except Exception as e:
            print(f"Could not load from {repo}: {e}")
            continue

    print("No valid checkpoint found. Starting fresh.")
    return None

def generate_sample(model, diffusion, epoch, output_dir, num_samples=1, ddim_steps=50):
    from diffusers import AutoencoderKL
    
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate one sample for each class (assuming 5 classes)
    classes_to_gen = [0, 1, 2, 3, 4] 
    
    with torch.no_grad():
        for class_idx in classes_to_gen:
            B = 1
            shape = (B, 4, 16, 32, 32)
            
            # Conditional Sampling
            y = torch.tensor([class_idx], device=device)
            
            # Use CFG=1.0 for training previews (faster), or 4.0 to see "real" quality
            z = diffusion.ddim_sample(
                model, shape, 
                y=y, cfg_scale=4.0, 
                ddim_steps=ddim_steps,
                device=device, progress=False
            )
            
            # Unscale latents (CRITICAL FIX)
            z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * 16, 4, 32, 32)
            z_flat = z_flat / 0.18215 
            
            video_flat = vae.decode(z_flat).sample
            video = video_flat.reshape(B, 16, 3, 32*8, 32*8).permute(0, 2, 1, 3, 4)
            video = torch.clamp((video + 1.0) / 2.0, 0, 1)
            
            video_np = (video[0].permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
            frames = [video_np[t] for t in range(16)]
            
            out_path = os.path.join(output_dir, f'epoch_{epoch:03d}_class_{class_idx}.mp4')
            imageio.mimwrite(out_path, frames, fps=8, codec='libx264', quality=8)
            print(f"Saved: {out_path}")
    
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dir', type=str, default='latents')
    parser.add_argument('--out_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--sample_dir', type=str, default='samples_v2')
    parser.add_argument('--epochs', type=int, default=2000) # Increased for small dataset
    parser.add_argument('--batch_size', type=int, default=16) # Increased for Blackwell GPU
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--sample_every', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=8)  # Higher for H100
    parser.add_argument('--compile', action='store_true', help='Use torch.compile for faster training')
    parser.add_argument('--grad_accum', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--fresh_start', action='store_true')
    
    # Model config
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--dim_head', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=5) # Default to 5 for your dataset
    
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    print(f"Loading latents from: {args.latent_dir}")
    dataset = LatentDataset(latent_dir=args.latent_dir)
    print(f"Found {len(dataset)} video clips.")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=4 if args.num_workers > 0 else None,  # Prefetch more batches
    )
    
    model = VideoDiTV2(
        in_channels=4,
        T=16, H=32, W=32,
        patch_size=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=args.dim_head,
        dropout=0.0,
        num_classes=args.num_classes
    ).to(device)

    # torch.compile for significant speedup (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile (this may take a few minutes on first run)...")
        model = torch.compile(model, mode="reduce-overhead")

    diffusion = GaussianDiffusion(timesteps=args.timesteps, beta_schedule='cosine')
    
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(dataloader), eta_min=1e-6
    )
    
    ema = EMA(model, decay=0.9999, warmup_steps=2000)
    scaler = GradScaler()

    start_epoch = 0
    global_step = 0

    if not args.fresh_start:
        checkpoint = get_latest_checkpoint()
        if checkpoint:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            ema.load_state_dict(checkpoint["ema"])
            start_epoch = checkpoint["epoch"] + 1
            global_step = checkpoint.get("step", 0)

    print(f"Starting training on {device}...")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for step, (latents, labels) in enumerate(pbar):
            global_step += 1
            latents = latents.to(device)
            labels = labels.to(device)

            t = torch.randint(0, args.timesteps, (latents.shape[0],), device=device)

            # --- CFG DROPOUT LOGIC ---
            # 10% chance to drop label to Null (index = num_classes)
            if np.random.random() < 0.1:
                y = torch.full_like(labels, model.num_classes)
            else:
                y = labels
            # -------------------------

            with autocast():
                loss = diffusion.training_losses(model, latents, t, y=y)
                loss = loss / args.grad_accum  # Scale loss for accumulation

            scaler.scale(loss).backward()

            # Only step optimizer after accumulating gradients
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                ema.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            if global_step % SAVE_EVERY_STEPS == 0:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                ckpt_name = f"checkpoint-step{global_step}-epoch{epoch}.pt"
                state = {
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "ema": ema.state_dict(),
                }
                local_path = f"/tmp/{ckpt_name}"
                torch.save(state, local_path)
                try:
                    upload_file(
                        path_or_fileobj=local_path,
                        path_in_repo=ckpt_name,
                        repo_id=REPO_ID,
                        commit_message=f"step {global_step}"
                    )
                    print(f"Pushed to HF")
                except Exception as e:
                    print(f"HF Upload Failed: {e}")
                if os.path.exists(local_path): os.remove(local_path)
        
        # Save local epoch checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.out_dir, f"epoch_{epoch+1}.pt")
            torch.save(ema.state_dict(), save_path)  # Save EMA weights for inference

        # Save full checkpoint to HuggingFace every N epochs (for spot instance recovery)
        if (epoch + 1) % HF_CHECKPOINT_EVERY_EPOCHS == 0:
            ckpt_name = f"checkpoint-epoch{epoch+1}.pt"
            state = {
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "ema": ema.state_dict(),
            }
            local_path = f"/tmp/{ckpt_name}"
            torch.save(state, local_path)
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=ckpt_name,
                    repo_id=REPO_ID,
                    commit_message=f"Epoch {epoch+1} checkpoint"
                )
                print(f"Epoch {epoch+1} checkpoint pushed to HuggingFace")
            except Exception as e:
                print(f"HF Upload Failed: {e}")
            if os.path.exists(local_path):
                os.remove(local_path)

        if (epoch + 1) % args.sample_every == 0:
            ema.apply_shadow()
            try:
                generate_sample(model, diffusion, epoch+1, args.sample_dir)
            except Exception as e:
                print(f"Sampling failed: {e}")
            ema.restore()

if __name__ == "__main__":
    main()