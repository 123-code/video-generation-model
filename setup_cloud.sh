#!/bin/bash
set -e

echo "=========================================="
echo "Cloud GPU Setup Script"
echo "=========================================="

echo "1. Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y unrar wget git

echo "2. Creating Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found. Please install Python 3.8+ first."
    exit 1
fi

python3 -m pip install --upgrade pip

echo "3. Installing Python dependencies..."
pip install -r requirements.txt

echo "4. Setting up HMDB51 dataset..."
bash setup_dataset.sh

echo "5. Verifying GPU availability..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Generate latents: python generate_latents.py"
echo "2. Train model: cd model && python train_video_dit.py --amp --batch_size 4"
echo ""

