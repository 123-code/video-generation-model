# Video Generation with VideoDiT

3D video generation using Diffusion Transformer (DiT) with temporal and spatial attention.

## Quick Start (Cloud GPU)

```bash
bash setup_cloud.sh
python generate_latents.py
cd model && python train_video_dit.py --amp --batch_size 4
```

**See `CLOUD_SETUP.md` for detailed cloud deployment guide.**

## Project Structure

```
video-generation-model/
├── model/
│   ├── video_dit.py           # VideoDiT model
│   ├── train_video_dit.py     # Training script
│   ├── vae.py                 # 3D VAE
│   ├── attention.py           # Spatial/temporal attention
│   └── ...
├── data/hmdb51/               # Video dataset (51 action categories)
├── latents/                   # Pre-computed VAE latents
├── generate_latents.py        # Generate latents from videos
├── generate_video.py          # Generate videos from trained model
├── setup_cloud.sh             # One-command cloud setup
├── setup_dataset.sh           # Download HMDB51 dataset
└── CLOUD_SETUP.md            # Detailed cloud guide
```

## Features

- **3D Diffusion Transformer** with spatiotemporal attention
- **Mixed precision training** (AMP) for faster training
- **Automatic checkpoint saving** every N epochs
- **Sample video generation** during training
- **HMDB51 dataset** support (51 action categories)
- **Cloud GPU ready** with one-command setup

## Training

### Basic
```bash
cd model
python train_video_dit.py
```

### Recommended (Cloud GPU)
```bash
cd model
python train_video_dit.py \
  --amp \
  --batch_size 4 \
  --epochs 50 \
  --save_every 5 \
  --generate_every 10 \
  --dim 768 \
  --depth 8
```

### Parameters
- `--amp`: Enable mixed precision training
- `--batch_size`: Batch size (2-8 depending on GPU)
- `--epochs`: Number of training epochs
- `--dim`: Model dimension (512/768/1024)
- `--depth`: Number of transformer layers
- `--save_every`: Save checkpoint every N epochs
- `--generate_every`: Generate sample video every N epochs

## Generation

```bash
python generate_video.py \
  --dit_checkpoint video_dit_checkpoint_epoch_50.pth \
  --vae_checkpoint vae_checkpoint_epoch_24.pth \
  --num_samples 4 \
  --steps 50
```

## Dataset Setup

### Automatic
```bash
bash setup_dataset.sh
```

### Python Script
```bash
python download_hmdb51.py
```

**Note:** Dataset is downloaded from Hugging Face. If it fails, login first:
```bash
huggingface-cli login
```

## Requirements

```
torch>=2.0.0
einops>=0.7.0
opencv-python>=4.8.0
imageio>=2.31.0
tqdm>=4.65.0
lpips>=0.1.4
datasets>=2.14.0
huggingface-hub>=0.17.0
```

Install: `pip install -r requirements.txt`

## Architecture

**VideoDiT** combines:
- 3D patch embedding for video tokens
- Spatial attention (within frames)
- Temporal attention (across frames)
- Diffusion denoising process
- 3D VAE for encoding/decoding

**Training process:**
1. Encode videos to latent space (3D VAE)
2. Add noise at random timesteps
3. Train model to predict noise
4. Generate by iterative denoising from random noise

## GPU Requirements

| GPU | VRAM | Batch Size | Model Size |
|-----|------|------------|------------|
| RTX 4060 Ti | 16GB | 2 | dim=512, depth=6 |
| RTX 3090 | 24GB | 4 | dim=768, depth=8 |
| A100 | 40GB+ | 8 | dim=1024, depth=12 |

## Training Time Estimates

- Small model (dim=512): ~4-6 hours (20 epochs, RTX 3090)
- Medium model (dim=768): ~8-12 hours (20 epochs, RTX 3090)
- Large model (dim=1024): ~16-24 hours (20 epochs, A100)

## Cloud Providers

- **RunPod**: ~$0.30/hr (RTX 3090)
- **Vast.ai**: ~$0.20/hr (RTX 3090)
- **Lambda Labs**: ~$0.50/hr (A100)
- **Google Colab Pro+**: $50/month

## Monitoring

```bash
# GPU usage
watch -n 1 nvidia-smi

# Training progress
ls -lh video_dit_checkpoint_*.pth
ls -lh generated_videos/

# View logs (if running in background)
tail -f train.log
```

## Troubleshooting

**Out of Memory:**
- Reduce `--batch_size`
- Reduce `--dim` and `--depth`
- Enable `--amp`

**Slow training:**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Enable `--amp` for mixed precision
- Use larger batch size if memory allows

**Poor quality:**
- Train for more epochs (50-100)
- Increase model size (`--dim 1024`)
- Generate with more steps (`--steps 100`)

## Citation

Based on:
- DiT (Diffusion Transformer)
- 3D VAE for video encoding
- HMDB51 dataset for human action recognition

