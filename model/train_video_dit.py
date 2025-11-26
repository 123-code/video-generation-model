import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
from tqdm import tqdm


FILE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.video_dit import VideoDiT
from diffusion import DiffusionProcess
from latent_dataset import LatentDataset
import imageio
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sample_video(model, diffusion, epoch, num_samples=1, steps=25):
    """Generate a sample video during training"""
    from diffusers import AutoencoderKL

    model.eval()
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Import the extract function
    from diffusion import extract

    with torch.no_grad():
        # Generate random latent
        B, C, T, H, W = num_samples, 4, 16, 32, 32  # Match our training latents
        z = torch.randn(B, C, T, H, W).to(device) * 0.18215  # Scaled latents

        # Reverse diffusion
        for i in range(steps, 0, -1):
            t = torch.full((B,), i-1, device=device, dtype=torch.long)
            pred_noise = model(z, t)

            # Use the extract function to properly handle device transfers
            beta_t = extract(diffusion.betas, t, z.shape)
            alpha_t = extract(diffusion.sqrt_alphas_cumprod, t, z.shape) ** 2  # alpha_t = 1 - beta_t
            alpha_bar_t = extract(diffusion.alphas_cumprod, t, z.shape)

            if t[0] > 0:
                noise = torch.randn_like(z)
                alpha_bar_prev = extract(diffusion.alphas_cumprod_prev, t, z.shape)
            else:
                noise = torch.zeros_like(z)
                alpha_bar_prev = torch.ones_like(alpha_bar_t)

            z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            if t[0] > 0:
                sigma_t = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                z = z + sigma_t * noise

        # Decode to video using VAE
        z_unscaled = z / 0.18215
        B, C, T, H, W = z_unscaled.shape
        z_flat = z_unscaled.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        video_flat = vae.decode(z_flat).sample
        video = video_flat.reshape(B, T, 3, H*8, W*8).permute(0, 2, 1, 3, 4)
        video = torch.clamp(video, -1.0, 1.0)
        video = (video + 1.0) / 2.0

        # Save as GIF for quick viewing
        video_np = video[0].cpu().numpy()  # Take first sample
        video_np = (video_np * 255).astype(np.uint8)
        C, T, H, W = video_np.shape

        frames = []
        for t in range(T):
            frame = video_np[:, t, :, :]
            frame = np.transpose(frame, (1, 2, 0))
            frames.append(frame)

        output_dir = "training_samples"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'sample_epoch_{epoch:02d}.gif')
        imageio.mimsave(output_path, frames, fps=4, loop=0)
        print(f"💾 Saved sample video: {output_path}")

    model.train()
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Train VideoDiT on Pre-computed SD Latents")
    parser.add_argument('--latent_dir', type=str, default=os.path.join(PROJECT_ROOT, 'latents'))
    parser.add_argument('--out_dir', type=str, default=PROJECT_ROOT)
    

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--generate_every', type=int, default=3, help="Generate sample every N epochs")
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--amp', action='store_true', help="Use Mixed Precision")
    

    parser.add_argument('--in_channels', type=int, default=4) 
    parser.add_argument('--T', type=int, default=16)          
    parser.add_argument('--H', type=int, default=32)          
    parser.add_argument('--W', type=int, default=32)          
    parser.add_argument('--dim', type=int, default=768)       
    parser.add_argument('--depth', type=int, default=12)      
    parser.add_argument('--heads', type=int, default=12)      
    
    args = parser.parse_args()

    print(f"Loading latents from: {args.latent_dir}")
    dataset = LatentDataset(latent_dir=args.latent_dir)
    print(f"Found {len(dataset)} video clips.")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}") 

    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    model = VideoDiT(
        in_channels=args.in_channels,
        T=args.T,
        H=args.H,
        W=args.W,
        patch_size=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads
    ).to(device)
    
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    diffusion = DiffusionProcess(timesteps=args.timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = amp.GradScaler(enabled=args.amp)
    criterion = nn.MSELoss()
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for step, latents in enumerate(pbar):
            latents = latents.to(device)
            
            t = torch.randint(0, args.timesteps, (latents.shape[0],), device=device).long()
            
            with amp.autocast(enabled=args.amp):
                noise = torch.randn_like(latents)
                x_t = diffusion.q_sample(latents, t, noise)
                predicted_noise = model(x_t, t)
                loss = criterion(predicted_noise, noise)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.6f}")

        if (epoch + 1) % args.save_every == 0:
            os.makedirs(args.out_dir, exist_ok=True)
            save_path = os.path.join(args.out_dir, f"video_dit_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

        if (epoch + 1) % args.generate_every == 0:
            print(f"🎬 Generating sample video at epoch {epoch+1}...")
            try:
                sample_path = generate_sample_video(model, diffusion, epoch+1)
                print(f"✅ Sample generated: {sample_path}")
            except Exception as e:
                print(f"❌ Sample generation failed: {e}")

if __name__ == "__main__":
    main()