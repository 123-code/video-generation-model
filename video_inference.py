import os 
import torch 
import torch.nn as nn 
import numpy as np
from tqdm import tqdm 
from model.dit import DiT
from model.vae import VAE3D
import cv2 

def generate_videos(dit_checkpoint,vae_checkpoint,output_dir,num_samples=5,timesteps=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = {
        'down_channels': [32, 64, 128],
        'mid_channels': [128, 128],
        'down_sample': [True, False],
        'num_down_layers': 1,
        'num_mid_layers': 1,
        'num_up_layers': 1,
        'attn_down': [False, False],
        'z_channels': 8,
        'norm_channels': 4,
        'num_heads': 1
    }
    #4 canales. uno mas que RGB, 4 capas en el dit block
    dit_model = DiT(channels=8,depth=4)
    dit_model.load_state_dict(torch.load(dit_checkpoint,map_location=device))
    dit_model.to(device)
    dit_model.eval()

    vae_model = VAE3D(im_channels=3, model_config=model_config).to(device)
    vae_model.load_state_dict(torch.load(vae_checkpoint,map_location=device))
    vae_model.eval()
#crea un array tamaño timesteps con valores entre 0.0001 y 0.02
    betas = torch.linspace(0.0001,0.02,timesteps)
    #cuantifica cunato de la imagen original queda despues de aplicar noising 
    alphas = 1.0 - betas 
    #producto acumulado de alphas
    alphas_cumprod = torch.cumprod(alphas,dim=0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0-alphas_cumprod)


    with torch.no_grad():
        for i in range(5):
            x = torch.randn(1, 8, 16, 32, 32).to(device)
            for t in tqdm(reversed(range(100)), desc=f"generating_sample{i+1}/1"):
                t_tensor = torch.full((1,), t, dtype=torch.long, device=device)
                alpha_t = sqrt_alphas_cumprod[t_tensor]
                sigma_t = sqrt_one_minus_alphas_cumprod[t_tensor]
                if t % 10 == 0:
                    print(f"t={t}, alpha_t stats: min={alpha_t.min().item()}, max={alpha_t.max().item()}, mean={alpha_t.mean().item()}, std={alpha_t.std().item()}")
                    print(f"t={t}, sigma_t stats: min={sigma_t.min().item()}, max={sigma_t.max().item()}, mean={sigma_t.mean().item()}, std={sigma_t.std().item()}")
                if torch.isnan(alpha_t).any() or torch.isnan(sigma_t).any() or torch.isinf(alpha_t).any() or torch.isinf(sigma_t).any():
                    print(f"NaN or Inf detected in alpha_t or sigma_t at timestep {t}")
                    break
                noise_pred = dit_model(x, t_tensor)
                if t % 10 == 0:
                    print(f"t={t}, noise_pred stats: min={noise_pred.min().item()}, max={noise_pred.max().item()}, mean={noise_pred.mean().item()}, std={noise_pred.std().item()}")
                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    print(f"NaN or Inf detected in noise_pred at timestep {t}")
                    break
                x = (x - sigma_t * noise_pred) / alpha_t
                if torch.isnan(x).any() or torch.isinf(x).any():
                    print(f"NaN or Inf detected in x after update at timestep {t}")
                    break
                if t > 0:
                    x += torch.randn_like(x) * sqrt_one_minus_alphas_cumprod[t_tensor - 1]
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        print(f"NaN or Inf detected in x after noise add at timestep {t}")
                        break

            print('Latent stats:', x.min().item(), x.max().item(), x.mean().item(), x.std().item())
            video = vae_model.decode(x)
            video = torch.clamp(video, 0, 1)
            video = video.permute(0, 2, 3, 4, 1).cpu().numpy()
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f'generated_video_{i}.mp4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(out_path, fourcc, 30.0, (128, 128))
            for frame in video[0]:
                frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)  
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)  
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(cv2.resize(frame, (128, 128)))  
            out.release()
            print(f"Saved generated video to {out_path}")

if __name__ == "__main__":
    dit_checkpoint_path = "dit_checkpoints/model_epoch_67.pth"
    vae_checkpoint_path = "video-generation-model/vae_checkpoint_epoch_24.pth"
    output_dir = "generated_videos"
    generate_videos(dit_checkpoint_path, vae_checkpoint_path, output_dir)
    
