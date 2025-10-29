import torch
import torch.nn as nn
from model.vae import VAE3D
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_size=(128, 128), transform=None, max_videos=None):
        self.data_dir = data_dir
        self.video_files = []
        for entry in os.listdir(data_dir):
            entry_path = os.path.join(data_dir, entry)
            if os.path.isdir(entry_path):
                self.video_files.extend([os.path.join(entry_path, f) for f in os.listdir(entry_path) if f.endswith('.avi')])
            elif entry.endswith('.avi'):
                self.video_files.append(entry_path)

        # Limit number of videos if specified
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]

        self.num_frames = num_frames
        self.frame_size = frame_size
        self.transform = transform

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count < self.num_frames:
            indices = np.arange(frame_count)
            indices = np.linspace(0, frame_count - 1, frame_count, dtype=int)
        else:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                frame = frame / 255.0
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                if self.transform:
                    frame = self.transform(frame)
                frames.append(frame)
        cap.release()

        # Pad with last frame if not enough frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        video = torch.stack(frames, dim=1)  # Shape: [C, T, H, W]
        return video, video_path

def generate_latents(data_dir, output_dir, batch_size=4, max_videos=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # VAE model configuration
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

    # Initialize VAE model
    vae = VAE3D(im_channels=3, model_config=model_config).to(device)
    vae.load_state_dict(torch.load('vae_checkpoint_epoch_24.pth', map_location=device))
    vae.eval()

    # Create dataset and dataloader
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = VideoDataset(
        data_dir=data_dir,
        num_frames=16,
        frame_size=(128, 128),
        transform=transform,
        max_videos=max_videos
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(dataset)} videos in batches of {batch_size}")

    with torch.no_grad():
        for batch_idx, (videos, video_paths) in enumerate(tqdm(dataloader, desc="Generating latents")):
            videos = videos.to(device)

            # Encode to latents
            latents, _ = vae.encode(videos)

            # Save latents
            for i, video_path in enumerate(video_paths):
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                latent_path = os.path.join(output_dir, f"{video_name}.pt")
                torch.save(latents[i].cpu(), latent_path)

    print(f"✓ Latents saved to {output_dir}")
    print(f"✓ Total videos processed: {len(dataset)}")
    print(f"✓ Latent shape: [8, 16, 32, 32] (channels, time, height, width)")

if __name__ == "__main__":
    data_dir = "data/hmdb51"
    output_dir = "latents_test"
    batch_size = 1  # Process one video at a time
    max_videos = 3  # Start with just 3 videos

    generate_latents(data_dir, output_dir, batch_size, max_videos)
