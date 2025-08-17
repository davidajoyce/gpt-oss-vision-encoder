#!/bin/bash
# Quick start script for RunPod - just paste this into terminal

echo "🚀 GPT-OSS Vision Training Quick Start"
echo "======================================"

# 1. Clone repo
echo "📦 Cloning repository..."
git clone -b djoyce/vision-encoder-llava https://github.com/YOUR_USERNAME/gpt-oss-vision-encoder.git
cd gpt-oss-vision-encoder

# 2. Build Docker image
echo "🐳 Building Docker image..."
docker build -t gpt-oss-vision:latest .

# 3. Run training
echo "🎯 Starting training on 10K samples..."
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  -v $(pwd)/data:/workspace/data \
  gpt-oss-vision:latest \
  python train_quick_10k.py

echo "✅ Training started! Check progress in the logs."
echo "💾 Checkpoints will be saved to ./checkpoints/"