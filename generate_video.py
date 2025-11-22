import os
import sys
import argparse
import torch
import imageio
import numpy as np
from tqdm import tqdm

FILE_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(FILE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.video_dit import VideoDiT
from model.vae import VAE3D
from diffusion import DiffusionProcess

LATENT_SCALE_FACTOR = 2.874741

def generate_video_clip(dit_model, vae3d, diffusion, num_samples=1, steps=50, device='cuda'):
    dit_model.eval()
    vae3d.eval()
    with torch.no_grad():
        B, C, T, H, W = num_samples, 8, 16, 64, 64
        z = torch.randn(B, C, T, H, W).to(device)
        
        for i in tqdm(range(steps, 0, -1), desc="Generating"):
            t = torch.full((B,), i-1, device=device, dtype=torch.long)
            pred_noise = dit_model(z, t)
            
            beta_t = diffusion.betas[t].view(-1, 1, 1, 1, 1)
            alpha_t = (1.0 - diffusion.betas)[t].view(-1, 1, 1, 1, 1)
            alpha_bar_t = diffusion.alphas_cumprod[t].view(-1, 1, 1, 1, 1)
            
            if t[0] > 0:
                noise = torch.randn_like(z)
                alpha_bar_prev = diffusion.alphas_cumprod[t-1].view(-1, 1, 1, 1, 1)
            else:
                noise = 0
                alpha_bar_prev = 1.0
            
            z = (1.0 / torch.sqrt(alpha_t)) * (z - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * pred_noise)
            if t[0] > 0:
                sigma_t = torch.sqrt(beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
                z = z + sigma_t * noise
        
        # <<< CRITICAL: scale UP before decoding (opposite of training) >>>
        z_scaled = z * LATENT_SCALE_FACTOR
        video = vae3d.decode(z_scaled)
        video = torch.clamp(video, -1.0, 1.0)
        video = (video + 1.0) / 2.0
        video = torch.clamp(video, 0.0, 1.0)
        
    return video

def save_video(video_tensor, output_path, fps=8):
    video_np = video_tensor.cpu().numpy()
    video_np = (video_np * 255).astype(np.uint8)
    B, C, T, H, W = video_np.shape
    
    for b in range(B):
        frames = []
        for t in range(T):
            frame = video_np[b, :, t, :, :]
            frame = np.transpose(frame, (1, 2, 0))
            frames.append(frame)
        
        video_path = output_path.replace('.mp4', f'_{b}.mp4')
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Saved video to {video_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate videos using trained VideoDiT")
    parser.add_argument('--dit_checkpoint', type=str, required=True)
    parser.add_argument('--vae_checkpoint', type=str, default='vae_checkpoint_epoch_24.pth')
    parser.add_argument('--num_samples', type=int, default=2)
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--output_dir', type=str, default='generated_videos')
    parser.add_argument('--in_channels', type=int, default=8)
    parser.add_argument('--T', type=int, default=16)
    parser.add_argument('--H', type=int, default=64)
    parser.add_argument('--W', type=int, default=64)
    parser.add_argument('--dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--timesteps', type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    print(f"Loading DiT checkpoint from {args.dit_checkpoint}")
    dit.load_state_dict(torch.load(args.dit_checkpoint, map_location=device))
    
    vae_config = {
        'z_channels': 8,
        'down_channels': [32, 64, 128, 128],
        'mid_channels': [128, 128],
        'down_sample': [True, True, True],
        'attn_down': [False, False, False],
        'norm_channels': 32,
        'num_heads': 4,
        'num_down_layers': 2,
        'num_mid_layers': 2,
        'num_up_layers': 2,
    }
    
    vae3d = VAE3D(im_channels=3, model_config=vae_config).to(device)
    print(f"Loading VAE checkpoint from {args.vae_checkpoint}")
    vae3d.load_state_dict(torch.load(args.vae_checkpoint, map_location=device))
    
    diffusion = DiffusionProcess(timesteps=args.timesteps)
    
    print(f"Generating {args.num_samples} video(s) with {args.steps} steps...")
    video = generate_video_clip(dit, vae3d, diffusion, num_samples=args.num_samples, steps=args.steps, device=device)
    
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'generated.mp4')
    save_video(video, output_path)
    
    print("Generation complete!")

if __name__ == "__main__":
    main()

