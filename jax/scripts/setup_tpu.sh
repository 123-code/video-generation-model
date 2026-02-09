#!/bin/bash
# TPU VM provisioning script for VideoDiTV2 JAX training
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - A GCP project with TPU API enabled
#
# Usage:
#   bash scripts/setup_tpu.sh [PROJECT_ID] [ZONE]

set -e

PROJECT_ID="${1:-your-gcp-project}"
ZONE="${2:-us-central1-a}"
TPU_NAME="video-dit-v2"
TPU_TYPE="v5e-4"
RUNTIME_VERSION="v2-alpha-tpuv5"

echo "=== Creating TPU VM ==="
echo "  Project: $PROJECT_ID"
echo "  Zone: $ZONE"
echo "  TPU: $TPU_NAME ($TPU_TYPE)"

# Create TPU VM (preemptible for cost savings)
gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --accelerator-type="$TPU_TYPE" \
    --version="$RUNTIME_VERSION" \
    --preemptible

echo "=== TPU VM created ==="

# SSH into TPU and install dependencies
echo "=== Installing dependencies ==="
gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT_ID" \
    --zone="$ZONE" \
    --command="
        # Update pip
        pip install --upgrade pip

        # Install JAX for TPU
        pip install 'jax[tpu]>=0.4.35' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

        # Install other dependencies
        pip install flax>=0.10.0 optax>=0.2.4 orbax-checkpoint>=0.11.0
        pip install grain-nightly ml_collections
        pip install numpy huggingface-hub>=0.23.0 imageio>=2.34.0 tqdm>=4.66.0

        # Verify JAX sees TPU
        python3 -c 'import jax; print(f\"JAX devices: {jax.device_count()} x {jax.devices()[0].platform}\")'

        echo 'Setup complete!'
    "

echo ""
echo "=== Setup complete ==="
echo "To SSH into the TPU VM:"
echo "  gcloud compute tpus tpu-vm ssh $TPU_NAME --project=$PROJECT_ID --zone=$ZONE"
echo ""
echo "To upload your code:"
echo "  gcloud compute tpus tpu-vm scp --recurse ./jax $TPU_NAME:~/video-dit-jax/ --project=$PROJECT_ID --zone=$ZONE"
echo ""
echo "To start training:"
echo "  python3 training/train.py --latent_dir ~/latents_npy"
echo ""
echo "To delete the TPU VM when done:"
echo "  gcloud compute tpus tpu-vm delete $TPU_NAME --project=$PROJECT_ID --zone=$ZONE"
