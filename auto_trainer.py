import os
import time
import subprocess
import sys
from huggingface_hub import snapshot_download, login

# --- CONFIGURATION ---
HF_TOKEN = "" # Your HF token
REPO_ID = "Jnaranjo/video-dit-spot"
LOCAL_LATENT_DIR = "latents"

# Arguments for your training setup
TRAINING_ARGS = [
    "--latent_dir", LOCAL_LATENT_DIR,
    "--out_dir", "checkpoints_v2",
    "--epochs", "5000",
    "--batch_size", "32",
    "--save_every", "10",
    "--num_classes", "11",  # Your 11 classes: bird, cat, clouds, dog, dolphin, fish, horse, rabbit, snake, tiger, wolf
    "--lr", "1e-4",
    "--fresh_start"  # Always start from scratch
]

def download_data():
    """
    Downloads latent files from HF repo.
    """
    print("
📥 Downloading latent data..."    print(f"Repository: {REPO_ID}")
    
    try:
        login(token=HF_TOKEN)
        
        local_dir = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=LOCAL_LATENT_DIR,
            allow_patterns=["*.pt"],  # Only download latent files
            ignore_patterns=["checkpoint-*", "*.pth"],  # Skip checkpoints
            token=HF_TOKEN
        )
        print(f"✅ Data downloaded to: {local_dir}")
        return True
    except Exception as e:
        print(f"❌ Data download failed: {e}")
        return False

def run_training():
    """
    Runs the training script.
    """
    cmd = [sys.executable, "train_v2.py"] + TRAINING_ARGS
    
    print("
🚀 Starting Video Generation Training"    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd)
        
        if result.returncode != 0:
            print(f"❌ Training failed with exit code {result.returncode}")
            return False
        else:
            print("✅ Training completed successfully!")
            return True
            
    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        return False
    except Exception as e:
        print(f"❌ Error running training: {e}")
        return False

def main():
    """
    Complete training pipeline: download data + train
    """
    # Step 1: Download data if needed
    if not os.path.exists(LOCAL_LATENT_DIR) or len(os.listdir(LOCAL_LATENT_DIR)) == 0:
        print("🔄 No local data found, downloading from HuggingFace...")
        if not download_data():
            print("💥 Critical error: Cannot download data. Exiting.")
            return
    else:
        print(f"📁 Using existing data in {LOCAL_LATENT_DIR}")
    
    # Count and verify data
    latent_files = [f for f in os.listdir(LOCAL_LATENT_DIR) if f.endswith('.pt')]
    print(f"📊 Found {len(latent_files)} latent files")
    
    if len(latent_files) == 0:
        print("❌ No latent files found!")
        return
    
    # Step 2: Run training
    success = run_training()
    
    if success:
        print("\n🎉 Complete training pipeline finished successfully!")
        print("Your model checkpoints are saved in: checkpoints_v2/")
    else:
        print("\n💥 Training pipeline failed. Check logs above.")

if __name__ == "__main__":
    main()
