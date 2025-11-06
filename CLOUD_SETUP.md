# Cloud GPU Setup Guide

## Quick Start (One Command)

```bash
git clone <your-repo-url>
cd video-generation-model
chmod +x setup_cloud.sh setup_dataset.sh
bash setup_cloud.sh
```

## Manual Setup

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd video-generation-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download HMDB51 Dataset

**Option A: Automatic (recommended)**
```bash
chmod +x setup_dataset.sh
bash setup_dataset.sh
```

**Option B: Python directly**
```bash
python download_hmdb51.py
```

This will:
- Download HMDB51 from Hugging Face (~2GB)
- Organize into action categories
- Place in `data/hmdb51/` directory

**Note:** If download fails, you may need to login to Hugging Face:
```bash
pip install huggingface-hub
huggingface-cli login
# Then enter your token from https://huggingface.co/settings/tokens
```

### 4. Generate Latents (Optional - if not using pre-generated)
```bash
python generate_latents.py
```

### 5. Train VideoDiT

**Basic training:**
```bash
cd model
python train_video_dit.py
```

**Optimized for cloud GPU:**
```bash
cd model
python train_video_dit.py \
  --epochs 50 \
  --batch_size 8 \
  --lr 1e-4 \
  --save_every 5 \
  --generate_every 10 \
  --amp \
  --latent_dir ../latents \
  --vae_checkpoint ../vae_checkpoint_epoch_24.pth
```

### 6. Monitor Training

Training outputs:
- Checkpoints: `video_dit_checkpoint_epoch_X.pth` (every 5 epochs)
- Sample videos: `generated_videos/epoch_X.mp4` (every 10 epochs)
- Progress bar with loss in terminal

### 7. Generate Videos

After training:
```bash
python generate_video.py \
  --dit_checkpoint video_dit_checkpoint_epoch_50.pth \
  --vae_checkpoint vae_checkpoint_epoch_24.pth \
  --num_samples 4 \
  --steps 50 \
  --output_dir generated_videos
```

## Cloud Provider Specific Instructions

### RunPod / Vast.ai
```bash
# Start with PyTorch template
# In terminal:
cd /workspace
git clone <your-repo-url>
cd video-generation-model
bash setup_cloud.sh
```

### Google Colab
```python
# In a cell:
!git clone <your-repo-url>
%cd video-generation-model
!bash setup_cloud.sh

# Train:
%cd model
!python train_video_dit.py --amp --batch_size 4 --epochs 30
```

### AWS/GCP/Azure
```bash
# SSH into instance
git clone <your-repo-url>
cd video-generation-model
bash setup_cloud.sh
cd model
nohup python train_video_dit.py --amp --batch_size 8 --epochs 50 > training.log 2>&1 &
tail -f training.log
```

## Training Configuration

### For 16GB GPU (e.g., Tesla T4, RTX 4060 Ti)
```bash
python train_video_dit.py \
  --batch_size 2 \
  --dim 512 \
  --depth 6 \
  --amp
```

### For 24GB GPU (e.g., RTX 3090, A5000)
```bash
python train_video_dit.py \
  --batch_size 4 \
  --dim 768 \
  --depth 8 \
  --amp
```

### For 40GB+ GPU (e.g., A100)
```bash
python train_video_dit.py \
  --batch_size 8 \
  --dim 1024 \
  --depth 12 \
  --amp
```

## File Structure After Setup

```
video-generation-model/
├── data/
│   └── hmdb51/              # Downloaded dataset
│       ├── brush_hair/
│       ├── cartwheel/
│       └── ... (51 categories)
├── latents/                 # Pre-computed VAE latents
├── model/
│   ├── train_video_dit.py   # Training script
│   └── ...
├── generated_videos/        # Generated samples
├── video_dit_checkpoint_*.pth  # Model checkpoints
└── vae_checkpoint_epoch_24.pth # VAE weights
```

## Troubleshooting

### Out of Memory
- Reduce `--batch_size`
- Reduce `--dim` (model dimension)
- Reduce `--depth` (number of layers)
- Ensure `--amp` is enabled

### Dataset Download Fails
```bash
# Login to Hugging Face first
huggingface-cli login

# Then try again
python download_hmdb51.py

# Or set token as environment variable
export HF_TOKEN=your_token_here
python download_hmdb51.py
```

### Missing VAE Checkpoint
Upload your local `vae_checkpoint_epoch_24.pth` to the cloud instance:
```bash
# From local machine:
scp vae_checkpoint_epoch_24.pth user@cloud-instance:/path/to/video-generation-model/
```

## Monitoring & Logs

### Real-time monitoring:
```bash
watch -n 1 nvidia-smi
```

### Check training progress:
```bash
ls -lh video_dit_checkpoint_*.pth
ls -lh generated_videos/
```

### Resume training from checkpoint:
Add to training script (modify train_video_dit.py):
```python
# Load checkpoint if exists
checkpoint_path = "video_dit_checkpoint_epoch_20.pth"
if os.path.exists(checkpoint_path):
    dit.load_state_dict(torch.load(checkpoint_path))
    print(f"Resumed from {checkpoint_path}")
```

## Estimated Training Time

- **Small model** (dim=512, depth=6): ~4-6 hours on RTX 3090 (20 epochs)
- **Medium model** (dim=768, depth=8): ~8-12 hours on RTX 3090 (20 epochs)
- **Large model** (dim=1024, depth=12): ~16-24 hours on A100 (20 epochs)

## Cost Estimates

- **RunPod RTX 3090**: ~$0.30/hr → $2.40 for 8 hours
- **Vast.ai RTX 3090**: ~$0.20/hr → $1.60 for 8 hours
- **AWS p3.2xlarge (V100)**: ~$3.06/hr → $24.48 for 8 hours
- **Google Colab Pro+**: $50/month unlimited

