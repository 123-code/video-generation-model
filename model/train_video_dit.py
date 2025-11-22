import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from tqdm import tqdm  # No longer needed for training progress
import torch.cuda.amp as amp
# import imageio  # No longer needed for video generation
# import numpy as np  # No longer needed for video generation

FILE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.video_dit import VideoDiT
# from model.vae import VAE3D  # No longer needed - switched to SD VAE
from diffusion import DiffusionProcess
from latent_dataset import LatentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 2
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_TIMESTEPS = 1000

# Video generation functions removed - now handled by generate_video_realism.py with SD VAE

def main():
    parser = argparse.ArgumentParser(description="Train VideoDiT on 3D video latents")
    parser.add_argument('--latent_dir', type=str, default=os.path.join(PROJECT_ROOT, 'latents'))
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--in_channels', type=int, default=4) # SD VAE has 4 channels
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=32)         # 256 / 8 = 32
    parser.add_argument('--W', type=int, default=32)         # 256 / 8 = 32
    parser.add_argument('--dim', type=int, default=768)      # Bigger dim for better quality
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--out_dir', type=str, default=PROJECT_ROOT)
    parser.add_argument('--vae_checkpoint', type=str, default=os.path.join(PROJECT_ROOT, 'vae_checkpoint_epoch_24.pth'))
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--generate_every', type=int, default=5)
    args = parser.parse_args()

    print("Loading latent dataset from:", args.latent_dir)
    dataset = LatentDataset(latent_dir=args.latent_dir)
    
    sample = dataset[0]
    print(f"Latent shape: {sample.shape}")
    
    dit = VideoDiT(
        in_channels=args.in_channels,
        T=args.T,
        H=args.H,
        W=args.W,
        patch_size=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads
    ).to(device)
    
    # Note: We switched to SD VAE, no longer using custom VAE3D for training
    # VAE is only needed for video generation, which is now handled by generate_video_realism.py
    
    diffusion = DiffusionProcess(timesteps=args.timesteps)
    optimizer = torch.optim.AdamW(dit.parameters(), lr=args.lr)
    scaler = amp.GradScaler(enabled=args.amp and torch.cuda.is_available())
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print(f"Model parameters: {sum(p.numel() for p in dit.parameters()):,}")
    print(f"Training on {len(dataset)} latents. Batch size: {args.batch_size}")
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    for epoch in range(args.epochs):
        dit.train()
        total_loss = 0
        print(f"Epoch {epoch+1}/{args.epochs}")

        for step, latents in enumerate(dataloader):
            latents = latents.to(device)
            
            if latents.dim() == 4:
                B, C, H, W = latents.shape
                latents = latents.unsqueeze(2).expand(B, C, args.T, H, W)
            
            # No division here. We trust the generation script.
            
            t = torch.randint(0, args.timesteps, (latents.shape[0],), device=device).long()
            
            with amp.autocast(enabled=args.amp and torch.cuda.is_available()):
                loss = diffusion.training_loss(dit, latents, t)
            
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0)
                optimizer.step()
            
            total_loss += loss.item()
            if step % 10 == 0:  # Print loss every 10 steps
                print(f"Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.epochs}, Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            ckpt_path = os.path.join(args.out_dir, f"video_dit_checkpoint_epoch_{epoch+1}.pth")
            torch.save(dit.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        
        # Skip video generation during training - use generate_video_realism.py instead
        # if (epoch + 1) % args.generate_every == 0:
        #     print(f"Generating sample video...")
        #     video = generate_video_clip(dit, vae3d, diffusion, num_samples=1, steps=50, device=device)
        #     gen_dir = os.path.join(args.out_dir, 'generated_videos')
        #     os.makedirs(gen_dir, exist_ok=True)
        #     video_path = os.path.join(gen_dir, f'epoch_{epoch+1}.mp4')
        #     save_video(video, video_path)
    
    print("Training complete!")

    # Skip final video generation - use generate_video_realism.py instead
    # print("Generating final video samples...")
    # video = generate_video_clip(dit, vae3d, diffusion, num_samples=2, steps=50, device=device)
    # gen_dir = os.path.join(args.out_dir, 'generated_videos')
    # os.makedirs(gen_dir, exist_ok=True)
    # video_path = os.path.join(gen_dir, 'final_samples.mp4')
    # save_video(video, video_path)

if __name__ == "__main__":
    main()

