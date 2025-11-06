#!/bin/bash
set -e

echo "=========================================="
echo "HMDB51 Dataset Setup Script"
echo "=========================================="

mkdir -p data
cd data

if [ -d "hmdb51" ] && [ "$(ls -A hmdb51)" ]; then
    echo "✓ HMDB51 dataset already exists"
    exit 0
fi

echo "Downloading HMDB51 dataset..."
wget http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar

echo "Installing unrar if needed..."
if ! command -v unrar &> /dev/null; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get update && sudo apt-get install -y unrar
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        brew install unrar
    fi
fi

echo "Extracting HMDB51 dataset..."
mkdir -p hmdb51
unrar x hmdb51_org.rar hmdb51/

cd hmdb51
echo "Extracting individual action categories..."
for file in *.rar; do
    if [ -f "$file" ]; then
        dirname="${file%.rar}"
        mkdir -p "$dirname"
        unrar x "$file" "$dirname/"
        rm "$file"
    fi
done

cd ../..

echo "Cleaning up..."
rm data/hmdb51_org.rar

echo "✓ HMDB51 dataset setup complete!"
echo "Dataset location: $(pwd)/data/hmdb51"
ls -d data/hmdb51/*/ | wc -l | xargs echo "Number of action categories:"

