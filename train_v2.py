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

from model.video_dit_v2 import VideoDiTV2
from diffusion_v2 import GaussianDiffusion, EMA
from latent_dataset import LatentDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sample(model, diffusion, epoch, output_dir, num_samples=1, ddim_steps=50):
    from diffusers import AutoencoderKL
    
    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        B, C, T, H, W = num_samples, 4, 16, 32, 32
        shape = (B, C, T, H, W)
        
        z = diffusion.ddim_sample(
            model, shape, 
            y=None, cfg_scale=1.0,
            ddim_steps=ddim_steps,
            device=device, progress=True
        )
        
        z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video_flat = vae.decode(z_flat).sample
        video = video_flat.reshape(B, T, 3, H*8, W*8).permute(0, 2, 1, 3, 4)
        video = torch.clamp((video + 1.0) / 2.0, 0, 1)
        
        for b in range(B):
            frames = []
            video_np = (video[b].permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
            for t in range(T):
                frames.append(video_np[t])
            
            out_path = os.path.join(output_dir, f'epoch_{epoch:03d}_sample_{b}.mp4')
            imageio.mimwrite(out_path, frames, fps=8, codec='libx264', quality=8)
            print(f"Saved: {out_path}")
    
    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latent_dir', type=str, default='latents')
    parser.add_argument('--out_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--sample_dir', type=str, default='samples_v2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--resume', type=str, default=None)
    
    # Model config for A6000
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--heads', type=int, default=16)
    parser.add_argument('--dim_head', type=int, default=64)
    
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
        drop_last=True
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
        num_classes=0
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    
    diffusion = GaussianDiffusion(timesteps=args.timesteps, beta_schedule='cosine')
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader),
        eta_min=1e-6
    )
    
    ema = EMA(model, decay=0.9999, warmup_steps=2000)
    scaler = GradScaler()
    
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        ema.load_state_dict(ckpt['ema'])
        start_epoch = ckpt['epoch'] + 1
    
    print(f"Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, latents in enumerate(pbar):
            latents = latents.to(device)
            
            t = torch.randint(0, args.timesteps, (latents.shape[0],), device=device)
            
            with autocast():
                loss = diffusion.training_losses(model, latents, t)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            scheduler.step()
            ema.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(args.out_dir, f"video_dit_v2_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'ema': ema.state_dict(),
                'loss': avg_loss
            }, save_path)
            print(f"Saved: {save_path}")
        
        if (epoch + 1) % args.sample_every == 0:
            print(f"Generating samples at epoch {epoch+1}...")
            ema.apply_shadow()
            try:
                generate_sample(model, diffusion, epoch+1, args.sample_dir)
            except Exception as e:
                print(f"Sample generation failed: {e}")
            ema.restore()

if __name__ == "__main__":
    main()

