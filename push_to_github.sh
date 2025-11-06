#!/bin/bash

echo "=========================================="
echo "Push to GitHub Helper Script"
echo "=========================================="

# Check if Git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing Git repository..."
    git init
fi

# Install Git LFS if not installed
if ! command -v git-lfs &> /dev/null; then
    echo ""
    echo "Git LFS is not installed. Installing..."
    brew install git-lfs  # macOS
    git lfs install
else
    echo "✓ Git LFS is installed"
fi

# Track large files
echo ""
echo "Setting up Git LFS for large files..."
git lfs track "*.pth"
git lfs track "*.pt"
git lfs track "*.mp4"

# Add files
echo ""
echo "Adding files to Git..."
git add .gitattributes
git add .

# Commit
echo ""
echo "Committing files..."
git commit -m "Add video generation training code"

# Ask for GitHub URL
echo ""
echo "=========================================="
echo "GitHub Setup"
echo "=========================================="
echo ""
echo "Go to GitHub and create a new repository:"
echo "https://github.com/new"
echo ""
echo "Then enter the repository URL below:"
echo "Example: https://github.com/yourusername/video-generation-model.git"
echo ""
read -p "GitHub URL: " GITHUB_URL

if [ -z "$GITHUB_URL" ]; then
    echo "No URL provided. Exiting."
    exit 1
fi

# Add remote
echo ""
echo "Adding remote..."
git remote remove origin 2>/dev/null
git remote add origin "$GITHUB_URL"

# Push
echo ""
echo "Pushing to GitHub..."
echo "This may take a while due to the large VAE checkpoint file..."
git push -u origin main --force

echo ""
echo "=========================================="
echo "✓ Done!"
echo "=========================================="
echo ""
echo "Your code is now on GitHub at:"
echo "$GITHUB_URL"
echo ""
echo "Next step: Update trainer.ipynb with your GitHub URL and upload to Kaggle!"

