# VideoDiT Training Guide

## Training the Model

### Basic Training
```bash
cd model
python train_video_dit.py
```

### Advanced Training Options
```bash
python train_video_dit.py \
  --latent_dir ../latents \
  --epochs 20 \
  --batch_size 2 \
  --lr 1e-4 \
  --save_every 5 \
  --generate_every 5 \
  --vae_checkpoint ../vae_checkpoint_epoch_24.pth \
  --amp
```

### Parameters
- `--latent_dir`: Directory containing pre-computed latents (default: ../latents)
- `--epochs`: Number of training epochs (default: 20)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--timesteps`: Diffusion timesteps (default: 1000)
- `--in_channels`: Latent channels (default: 8)
- `--T`: Temporal dimension (default: 16)
- `--H`: Height dimension (default: 64)
- `--W`: Width dimension (default: 64)
- `--dim`: Model dimension (default: 768)
- `--depth`: Number of transformer blocks (default: 6)
- `--heads`: Number of attention heads (default: 8)
- `--save_every`: Save checkpoint every N epochs (default: 5)
- `--generate_every`: Generate sample video every N epochs (default: 5)
- `--vae_checkpoint`: Path to VAE checkpoint
- `--amp`: Enable mixed precision training

## Generating Videos

After training, generate videos using:

```bash
python generate_video.py \
  --dit_checkpoint video_dit_checkpoint_epoch_20.pth \
  --vae_checkpoint vae_checkpoint_epoch_24.pth \
  --num_samples 2 \
  --steps 50 \
  --output_dir generated_videos
```

### Generation Parameters
- `--dit_checkpoint`: Path to trained VideoDiT checkpoint (required)
- `--vae_checkpoint`: Path to VAE checkpoint
- `--num_samples`: Number of videos to generate (default: 2)
- `--steps`: Number of diffusion steps (default: 50)
- `--output_dir`: Output directory for generated videos

## Architecture Overview

The VideoDiT uses:
- 3D patch embedding for spatiotemporal tokens
- Spatial and temporal attention blocks
- Diffusion-based denoising for video generation
- 3D VAE for latent space encoding/decoding

## Training Features

- Automatic checkpoint saving every 5 epochs
- Sample video generation during training
- Mixed precision training support (--amp)
- Gradient clipping for stability
- Progress bars with loss tracking
- Latent scaling for better training dynamics

