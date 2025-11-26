import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision import transforms

def generate_latents(data_dir, output_dir, batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load SD VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    
    # 2. Define Transforms (SD expects [-1, 1])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)), # HMDB51 is small, 256 is standard
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    os.makedirs(output_dir, exist_ok=True)
    
    # Walk through HMDB51 folders
    video_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.avi'):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} videos")

    with torch.no_grad():
        for video_path in tqdm(video_files):
            filename = os.path.splitext(os.path.basename(video_path))[0]
            save_path = os.path.join(output_dir, f"{filename}.pt")
            
            if os.path.exists(save_path): continue

            # Read Video using OpenCV
            cap = cv2.VideoCapture(video_path)
            frames = []
            while len(frames) < 16: # Target 16 frames
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(transform(frame))
            cap.release()

            # Pad if video is too short
            if len(frames) < 16 and len(frames) > 0:
                while len(frames) < 16:
                    frames.append(frames[-1])
            
            if len(frames) == 16:
                # Stack: [16, 3, 256, 256]
                pixel_values = torch.stack(frames).to(device)
                
                # Encode batch of images
                # Dist: [16, 4, 32, 32]
                latents = vae.encode(pixel_values).latent_dist.sample()
                
                # Scale immediately to save compute during training
                latents = latents * 0.18215
                
                # Rearrange to [4, 16, 32, 32] (Channels, Time, H, W) for your DiT
                latents = latents.permute(1, 0, 2, 3) 
                
                torch.save(latents.cpu(), save_path)

if __name__ == "__main__":
    generate_latents("data/hmdb51", "latents")