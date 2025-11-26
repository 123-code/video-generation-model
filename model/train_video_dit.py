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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser(description="Train VideoDiT on Pre-computed SD Latents")
    parser.add_argument('--latent_dir', type=str, default=os.path.join(PROJECT_ROOT, 'latents'))
    parser.add_argument('--out_dir', type=str, default=PROJECT_ROOT)
    

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4) 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--timesteps', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5)
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

if __name__ == "__main__":
    main()