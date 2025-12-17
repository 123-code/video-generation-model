import os
import time
import subprocess
import sys
from huggingface_hub import snapshot_download, login

# --- CONFIGURATION ---
HF_TOKEN = "xx" # Replace with your WRITE token
REPO_ID = "Jnaranjo/video-dit-spot"
LOCAL_LATENT_DIR = "latents_hq"

# Arguments to pass to train_v2.py
TRAINING_ARGS = [
    "--latent_dir", LOCAL_LATENT_DIR,
    "--out_dir", "checkpoints_v2",
    "--epochs", "5000",          # Set high, we stop when we want
    "--batch_size", "32",        # Adjust for Blackwell GPU
    "--save_every", "10",       # Save checkpoint every 100 epochs
    "--num_classes", "5",        # Critical for your Conditional model
    "--lr", "1e-4"
]

def download_data():
    """
    Downloads ONLY the latent files (.pt) from the HF repo.
    Skips downloading massive .pth checkpoint files to save time.
    """
    print(f"\n--- 1. Syncing Latents from {REPO_ID} ---")
    try:
        login(token=HF_TOKEN)
        
        # We assume latents are in the root or a folder in the repo.
        # This downloads everything ending in .pt
        # If your latents are in a subfolder in the repo, usage matches structure.
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset", # Change to "model" if you uploaded them to a model repo
            local_dir=LOCAL_LATENT_DIR,
            allow_patterns=["*.pt"], # Only download latents
            ignore_patterns=["checkpoint-*", "*.pth"], # Don't re-download old checkpoints
            token=HF_TOKEN
        )
        print(f"✅ Data synced to {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False

def run_training():
    """
    Runs the training script. If it crashes, this function returns False.
    """
    cmd = [sys.executable, "train_v2.py"] + TRAINING_ARGS
    
    print(f"\n--- 2. Starting Training ---")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the training script and wait for it
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"⚠️ Training crashed with error code {result.returncode}")
            return False
        else:
            print("✅ Training finished successfully.")
            return True
            
    except KeyboardInterrupt:
        print("\n🛑 Training stopped by user.")
        sys.exit(0)
    except Exception as e:
        print(f"⚠️ Error executing script: {e}")
        return False

def main_loop():
    # 1. Initial Data Sync
    if not download_data():
        print("Critical error downloading data. Exiting.")
        return

    # 2. Infinite Loop for Reliability
    restart_count = 0
    
    while True:
        success = run_training()
        
        if success:
            print("Training complete!")
            break
        
        restart_count += 1
        print(f"\n🔄 Crash detected! Restarting in 10 seconds... (Restart #{restart_count})")
        print("Note: The script will automatically resume from the last HF checkpoint.")
        time.sleep(10)

if __name__ == "__main__":
    main_loop()