import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from diffusers import AutoencoderKL
from torchvision import transforms

def generate_latents(data_dir, output_dir, batch_size=8):
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
    print(f"Using batch size: {batch_size}")

    # Filter out already processed videos
    videos_to_process = []
    for video_path in video_files:
        filename = os.path.splitext(os.path.basename(video_path))[0]
        save_path = os.path.join(output_dir, f"{filename}.pt")
        if not os.path.exists(save_path):
            videos_to_process.append(video_path)

    print(f"Videos to process: {len(videos_to_process)}")

    with torch.no_grad():
        # Process videos in batches
        for i in tqdm(range(0, len(videos_to_process), batch_size)):
            batch_videos = videos_to_process[i:i+batch_size]
            batch_frames = []
            valid_indices = []

            # Read and preprocess all videos in batch
            for j, video_path in enumerate(batch_videos):
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
                    batch_frames.extend(frames)  # Add all 16 frames
                    valid_indices.append(j)

            if not batch_frames:
                continue

            # Stack all frames: [batch_size * 16, 3, 256, 256]
            pixel_values = torch.stack(batch_frames).to(device)

            # Encode batch of images
            latents = vae.encode(pixel_values).latent_dist.sample()

            # Scale immediately to save compute during training
            latents = latents * 0.18215

            # Reshape back to [batch_size, 4, 16, 32, 32] and permute to [batch_size, 4, 16, 32, 32]
            # latents is currently [batch_size * 16, 4, 32, 32]
            num_videos = len(valid_indices)
            latents = latents.view(num_videos, 16, 4, 32, 32).permute(0, 2, 1, 3, 4)

            # Save each video's latents
            for j, video_idx in enumerate(valid_indices):
                video_path = batch_videos[video_idx]
                filename = os.path.splitext(os.path.basename(video_path))[0]
                save_path = os.path.join(output_dir, f"{filename}.pt")
                torch.save(latents[j].cpu(), save_path)

if __name__ == "__main__":
    generate_latents("hmdb51_root/hmdb51", "latents")