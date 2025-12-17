import os
import argparse
import torch
import numpy as np
import imageio
from tqdm import tqdm

from model.video_dit_v2 import VideoDiTV2
from diffusion_v2 import GaussianDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_vae():
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    return vae

def decode_latents(vae, z):
    B, C, T, H, W = z.shape
    z_flat = z.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    with torch.no_grad():
        video_flat = vae.decode(z_flat).sample
    video = video_flat.reshape(B, T, 3, H*8, W*8).permute(0, 2, 1, 3, 4)
    video = torch.clamp((video + 1.0) / 2.0, 0, 1)
    return video

def save_video(video, path, fps=8):
    video_np = (video.permute(1, 2, 3, 0).cpu().numpy() * 255).astype(np.uint8)
    frames = [video_np[t] for t in range(video_np.shape[0])]
    imageio.mimwrite(path, frames, fps=fps, codec='libx264', quality=8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='generated_v2')
    parser.add_argument('--num_samples', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--ddim_steps', type=int, default=50)
    parser.add_argument('--use_ddpm', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    # Model config (must match training)
    parser.add_argument('--dim', type=int, default=1024)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--heads', type=int, default=16)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading model...")
    model = VideoDiTV2(
        in_channels=4,
        T=16, H=32, W=32,
        patch_size=2,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        dim_head=64,
        dropout=0.0,
        num_classes=0
    ).to(device)
    
    ckpt = torch.load(args.checkpoint, map_location=device)
    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    print("Loading VAE...")
    vae = load_vae()
    
    diffusion = GaussianDiffusion(timesteps=1000, beta_schedule='cosine')
    
    print(f"Generating {args.num_samples} videos...")
    generated = 0
    
    with torch.no_grad():
        while generated < args.num_samples:
            batch_size = min(args.batch_size, args.num_samples - generated)
            shape = (batch_size, 4, 16, 32, 32)
            
            if args.use_ddpm:
                z = diffusion.ddpm_sample(model, shape, device=device)
            else:
                z = diffusion.ddim_sample(
                    model, shape,
                    ddim_steps=args.ddim_steps,
                    device=device
                )
            
            videos = decode_latents(vae, z)
            
            for i in range(batch_size):
                out_path = os.path.join(args.output_dir, f'generated_{generated}.mp4')
                save_video(videos[i], out_path)
                print(f"Saved: {out_path}")
                generated += 1
    
    print(f"Done! Generated {args.num_samples} videos in {args.output_dir}")

if __name__ == "__main__":
    main()
