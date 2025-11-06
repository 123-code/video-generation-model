#!/bin/bash
set -e

echo "=========================================="
echo "HMDB51 Dataset Setup Script"
echo "=========================================="

if [ -d "data/hmdb51" ] && [ "$(ls -A data/hmdb51 2>/dev/null)" ]; then
    echo "✓ HMDB51 dataset already exists"
    exit 0
fi

echo "Downloading HMDB51 from Hugging Face..."
echo "This uses the datasets library (via Python)"
echo ""

python3 download_hmdb51.py

echo ""
echo "✓ HMDB51 dataset setup complete!"

