"""
Complete training pipeline:
1. Download videos from Pexels (or use existing)
2. Generate VAE latents (or use existing)
3. Resume training from HuggingFace checkpoint
"""
import os
import subprocess
import sys
from pathlib import Path

# --- CONFIGURATION ---
PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Directories
VIDEO_DIR = "videos"
LATENT_DIR = "latents"
CHECKPOINT_DIR = "checkpoints_v2"

# Categories (11 classes)
CATEGORIES = [
    "bird", "cat", "clouds", "dog", "dolphin",
    "fish", "horse", "rabbit", "snake", "tiger", "wolf"
]

VIDEOS_PER_CATEGORY = 1500  # Target ~14k total videos

# Training arguments
TRAINING_ARGS = [
    "--latent_dir", LATENT_DIR,
    "--out_dir", CHECKPOINT_DIR,
    "--epochs", "5000",
    "--batch_size", "32",
    "--num_classes", "11",
    "--lr", "1e-4",
    "--compile",  # Enable torch.compile for faster training
    "--num_workers", "8",
    # Note: NOT using --fresh_start so it resumes from checkpoint
]


def count_files(directory: str, extension: str = ".pt") -> int:
    """Count files with given extension in directory (recursive)."""
    path = Path(directory)
    if not path.exists():
        return 0
    return len(list(path.rglob(f"*{extension}")))


def download_pexels_videos():
    """Download videos from Pexels API."""
    if not PEXELS_API_KEY:
        print("PEXELS_API_KEY not set. Skipping Pexels download.")
        print("Set it with: export PEXELS_API_KEY='your-api-key'")
        return False

    existing_videos = count_files(VIDEO_DIR, ".mp4")
    target_videos = len(CATEGORIES) * VIDEOS_PER_CATEGORY

    if existing_videos >= target_videos * 0.9:  # 90% threshold
        print(f"Found {existing_videos} videos, skipping download.")
        return True

    print(f"\nDownloading videos from Pexels...")
    print(f"Target: {target_videos} videos ({VIDEOS_PER_CATEGORY} per category)")

    cmd = [
        sys.executable, "download_pexels.py",
        "--output_dir", VIDEO_DIR,
        "--videos_per_category", str(VIDEOS_PER_CATEGORY),
        "--categories", *CATEGORIES,
    ]

    env = os.environ.copy()
    env["PEXELS_API_KEY"] = PEXELS_API_KEY

    result = subprocess.run(cmd, env=env)
    return result.returncode == 0


def generate_latents():
    """Generate VAE latents from videos."""
    existing_latents = count_files(LATENT_DIR, ".pt")
    existing_videos = count_files(VIDEO_DIR, ".mp4")

    if existing_latents >= existing_videos * 0.9 and existing_latents > 0:
        print(f"Found {existing_latents} latents, skipping generation.")
        return True

    if existing_videos == 0:
        print("No videos found! Run Pexels download first or add videos to 'videos/' directory.")
        return False

    print(f"\nGenerating VAE latents from {existing_videos} videos...")

    cmd = [
        sys.executable, "generate_latents.py",
        "--input_dir", VIDEO_DIR,
        "--output_dir", LATENT_DIR,
    ]

    result = subprocess.run(cmd)
    return result.returncode == 0


def download_existing_latents():
    """Download pre-computed latents from HuggingFace if available."""
    from huggingface_hub import snapshot_download, login

    print("\nAttempting to download pre-computed latents from HuggingFace...")

    try:
        if HF_TOKEN:
            login(token=HF_TOKEN)

        local_dir = snapshot_download(
            repo_id="Jnaranjo/video-dit-spot",
            repo_type="dataset",
            local_dir=LATENT_DIR,
            allow_patterns=["*.pt"],
            ignore_patterns=["checkpoint-*", "*.pth"],
            token=HF_TOKEN if HF_TOKEN else None,
        )
        print(f"Downloaded latents to: {local_dir}")
        return True
    except Exception as e:
        print(f"Could not download latents: {e}")
        return False


def run_training():
    """Run the training script with resume from checkpoint."""
    latent_count = count_files(LATENT_DIR, ".pt")
    if latent_count == 0:
        print("No latent files found! Cannot start training.")
        return False

    print(f"\nStarting training with {latent_count} latent files...")
    print(f"Will resume from HuggingFace checkpoint (epoch 155)")
    print("=" * 60)

    cmd = [sys.executable, "train_v2.py"] + TRAINING_ARGS

    try:
        result = subprocess.run(cmd)
        return result.returncode == 0
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return False
    except Exception as e:
        print(f"Training error: {e}")
        return False


def main():
    """Complete pipeline: data preparation + training."""
    print("=" * 60)
    print("Video Generation Model - Complete Training Pipeline")
    print("=" * 60)

    # Step 1: Check/prepare data
    latent_count = count_files(LATENT_DIR, ".pt")
    video_count = count_files(VIDEO_DIR, ".mp4")

    print(f"\nCurrent state:")
    print(f"  Videos: {video_count}")
    print(f"  Latents: {latent_count}")

    if latent_count < 1000:  # Need more data
        print("\nNeed to prepare training data...")

        # Try downloading pre-computed latents first (fastest)
        if download_existing_latents():
            latent_count = count_files(LATENT_DIR, ".pt")
            print(f"Now have {latent_count} latents.")

        # If still not enough, download videos and generate latents
        if latent_count < 1000:
            print("\nDownloading videos from Pexels...")
            if download_pexels_videos():
                print("\nGenerating latents from videos...")
                generate_latents()
            else:
                print("Could not download videos. Please:")
                print("1. Set PEXELS_API_KEY environment variable, or")
                print("2. Manually add videos to 'videos/<category>/' directories")

    # Step 2: Run training
    latent_count = count_files(LATENT_DIR, ".pt")
    if latent_count > 0:
        print(f"\nReady to train with {latent_count} video clips")
        success = run_training()
        if success:
            print("\nTraining completed successfully!")
        else:
            print("\nTraining stopped. Checkpoints saved to HuggingFace for resume.")
    else:
        print("\nNo training data available. Please prepare data first.")


if __name__ == "__main__":
    main()
