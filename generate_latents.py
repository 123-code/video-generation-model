"""
Generate VAE latents from video files for training.
Processes videos into 16-frame clips encoded via SD VAE.
"""
import os
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from diffusers import AutoencoderKL
import argparse


def extract_frames(video_path: str, num_frames: int = 16, target_size: int = 256):
    """Extract and preprocess frames from a video."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < num_frames:
        cap.release()
        return None

    # Sample frames evenly across the video
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize and center crop
        h, w = frame.shape[:2]
        scale = target_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Center crop
        start_h = (new_h - target_size) // 2
        start_w = (new_w - target_size) // 2
        frame = frame[start_h : start_h + target_size, start_w : start_w + target_size]

        frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        return None

    # Convert to tensor: [T, H, W, C] -> [T, C, H, W]
    frames = np.stack(frames)
    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()

    # Normalize to [-1, 1]
    frames = (frames / 127.5) - 1.0

    return frames


def encode_video(frames: torch.Tensor, vae: AutoencoderKL, device: str):
    """Encode video frames to VAE latents."""
    # frames: [T, C, H, W]
    T = frames.shape[0]

    with torch.no_grad():
        frames = frames.to(device)
        # Encode all frames at once
        latents = vae.encode(frames).latent_dist.sample()
        # Scale latents (SD VAE scaling factor)
        latents = latents * 0.18215

    # Reshape: [T, 4, 32, 32] -> [4, T, 32, 32]
    latents = latents.permute(1, 0, 2, 3)

    return latents.cpu()


def process_videos(
    input_dir: str,
    output_dir: str,
    batch_size: int = 8,
    num_frames: int = 16,
    target_size: int = 256,
):
    """Process all videos in input directory to latents."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print("Loading VAE...")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Get all video files organized by category
    categories = [d for d in input_path.iterdir() if d.is_dir()]

    if not categories:
        # Flat directory structure
        categories = [input_path]
        is_flat = True
    else:
        is_flat = False

    total_processed = 0
    total_failed = 0

    for category_dir in categories:
        if is_flat:
            category_name = "uncategorized"
            video_files = list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.avi"))
        else:
            category_name = category_dir.name
            video_files = list(category_dir.glob("*.mp4")) + list(category_dir.glob("*.avi"))

        if not video_files:
            continue

        # Create output directory
        category_output = output_path / category_name
        category_output.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing category: {category_name} ({len(video_files)} videos)")

        pbar = tqdm(video_files, desc=category_name)
        for video_file in pbar:
            output_file = category_output / f"{video_file.stem}.pt"

            if output_file.exists():
                total_processed += 1
                continue

            frames = extract_frames(str(video_file), num_frames, target_size)
            if frames is None:
                total_failed += 1
                continue

            latents = encode_video(frames, vae, device)
            torch.save(latents, output_file)
            total_processed += 1
            pbar.set_postfix(processed=total_processed, failed=total_failed)

    print(f"\nProcessing complete!")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")
    print(f"Latents saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate VAE latents from videos")
    parser.add_argument("--input_dir", type=str, default="videos", help="Input video directory")
    parser.add_argument("--output_dir", type=str, default="latents", help="Output latent directory")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--target_size", type=int, default=256)
    args = parser.parse_args()

    process_videos(
        args.input_dir,
        args.output_dir,
        args.batch_size,
        args.num_frames,
        args.target_size,
    )


if __name__ == "__main__":
    main()
