#!/bin/bash
set -e

echo "=========================================="
echo "Video Generation Model - Quick Start"
echo "=========================================="

if [ ! -f "vae_checkpoint_epoch_24.pth" ]; then
    echo "⚠️  WARNING: vae_checkpoint_epoch_24.pth not found!"
    echo "Please upload your VAE checkpoint before training."
    read -p "Do you have the VAE checkpoint ready? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please add VAE checkpoint and run again."
        exit 1
    fi
fi

if [ ! -d "data/hmdb51" ] || [ -z "$(ls -A data/hmdb51)" ]; then
    echo "Setting up HMDB51 dataset..."
    bash setup_dataset.sh
else
    echo "✓ HMDB51 dataset found"
fi

if [ ! -d "latents" ] || [ -z "$(ls -A latents)" ]; then
    echo ""
    echo "No pre-computed latents found."
    read -p "Generate latents now? This will take some time. (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python generate_latents.py
    else
        echo "⚠️  You'll need latents before training. Run: python generate_latents.py"
    fi
else
    echo "✓ Latents found: $(ls latents/*.pt | wc -l) files"
fi

echo ""
echo "=========================================="
echo "Ready to train!"
echo "=========================================="
echo ""
echo "Start training with:"
echo "  cd model"
echo "  python train_video_dit.py --amp --batch_size 4"
echo ""
echo "Or with custom settings:"
echo "  python train_video_dit.py --epochs 50 --batch_size 8 --amp --generate_every 5"
echo ""

