from diffusers import AutoencoderKL
import torch
from torch.utils.data import DataLoader, Dataset
from moving_mnist_dataset import MovingMNISTDataset
import os
import cv2
import numpy as np
from tqdm import tqdm

class VideoDataset(Dataset):
    def __init__(self, data_dir, num_frames=16, frame_size=(64, 64), max_videos=None):
        self.data_dir = data_dir
        self.video_files = []

        # Find all video files in the directory
        for root, _, files in os.walk(data_dir):
            for fname in files:
                if fname.endswith(('.avi', '.mp4', '.mov')):
                    self.video_files.append(os.path.join(root, fname))

        # Limit number of videos if specified
        if max_videos is not None:
            self.video_files = self.video_files[:max_videos]

        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count < self.num_frames:
            indices = np.linspace(0, frame_count - 1, frame_count, dtype=int)
        else:
            indices = np.linspace(0, frame_count - 1, self.num_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.frame_size)
                # Convert to [-1, 1] range for SD VAE
                frame = frame.astype(np.float32) / 127.5 - 1.0
                frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)
                frames.append(frame)

        cap.release()

        # Pad with last frame if not enough frames
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())

        # Stack frames: [C, T, H, W]
        video = torch.stack(frames, dim=1)

        # Extract video name from path
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        return video, video_name

def encode_video(video_tensor, vae):
    """
    Encode video tensor using SD VAE
    Input: [B, 3, T, H, W]  (Batch, RGB, Time, Height, Width)
    Input should be normalized to [-1, 1]
    """
    B, C, T, H, W = video_tensor.shape

    # 1. Squash Batch and Time dimensions together
    # We pretend we have (B * T) individual images
    x = video_tensor.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)

    with torch.no_grad():
        # 2. Encode using SD VAE
        posterior = vae.encode(x).latent_dist
        latents = posterior.sample()

        # 3. Apply the Magic Scaling Factor
        # SD VAEs are trained with this factor. Without it, your training will fail.
        latents = latents * 0.18215

    # 4. Reshape back to Video format
    # SD VAE compresses spatial dims by 8 (e.g., 64x64 -> 8x8)
    # Output channels are 4
    _, C_out, H_out, W_out = latents.shape
    latents = latents.reshape(B, T, C_out, H_out, W_out).permute(0, 2, 1, 3, 4)

    return latents  # Output: [B, 4, T, H/8, W/8]

def generate_latents_from_videos(data_dir, output_dir, batch_size=4, max_videos=10, num_frames=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load standard SD VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Create video dataset
    dataset = VideoDataset(
        data_dir=data_dir,
        num_frames=num_frames,
        frame_size=(256, 256),
        max_videos=max_videos
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(dataset)} videos in batches of {batch_size}")

    with torch.no_grad():
        for batch_idx, (videos, video_names) in enumerate(tqdm(dataloader, desc="Generating latents")):
            videos = videos.to(device)

            # Encode to latents using SD VAE
            latents = encode_video(videos, vae)

            # Save latents
            for i, video_name in enumerate(video_names):
                latent_path = os.path.join(output_dir, f"{video_name}.pt")
                torch.save(latents[i].cpu(), latent_path)

    print(f"✓ Latents saved to {output_dir}")
    print(f"✓ Total videos processed: {len(dataset)}")
    print(f"✓ Latent shape: [4, {num_frames}, 32, 32] (channels, time, height, width)")

def generate_latents_from_mnist(output_dir, batch_size=4, num_samples=1000, seq_len=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load standard SD VAE
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    # Create Moving MNIST dataset
    dataset = MovingMNISTDataset(
        num_samples=num_samples,
        seq_len=seq_len,
        image_size=256,
        num_digits=2,
        step_length=0.1
    )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing {len(dataset)} videos in batches of {batch_size}")

    with torch.no_grad():
        for batch_idx, (videos, video_names) in enumerate(tqdm(dataloader, desc="Generating latents")):
            videos = videos.to(device)

            # Encode to latents using SD VAE
            latents = encode_video(videos, vae)

            # Save latents
            for i, video_name in enumerate(video_names):
                latent_path = os.path.join(output_dir, f"{video_name}.pt")
                torch.save(latents[i].cpu(), latent_path)

    print(f"✓ Latents saved to {output_dir}")
    print(f"✓ Total videos processed: {len(dataset)}")
    print(f"✓ Latent shape: [4, {seq_len}, 32, 32] (channels, time, height, width)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['mnist', 'videos'], default='mnist',
                       help='Generate latents from Moving MNIST or video files')
    parser.add_argument('--data_dir', type=str, help='Directory containing video files (for videos mode)')
    parser.add_argument('--output_dir', type=str, default="latents")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of MNIST samples')
    parser.add_argument('--seq_len', type=int, default=20, help='Sequence length for MNIST')
    parser.add_argument('--max_videos', type=int, default=10, help='Max videos to process')
    parser.add_argument('--num_frames', type=int, default=16, help='Number of frames per video')
    args = parser.parse_args()

    if args.mode == 'videos':
        if not args.data_dir:
            print("Error: --data_dir is required when using --mode videos")
            exit(1)
        generate_latents_from_videos(args.data_dir, args.output_dir, args.batch_size, args.max_videos, args.num_frames)
    else:
        generate_latents_from_mnist(args.output_dir, args.batch_size, args.num_samples, args.seq_len)
