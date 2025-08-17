#!/bin/bash
# Simple RunPod setup without Docker - just run this in RunPod terminal

echo "🚀 GPT-OSS Vision Training - No Docker Setup"
echo "============================================="

# 1. Update system and install git
echo "📦 Installing dependencies..."
apt-get update -qq && apt-get install -y git wget

# 2. Clone repository
echo "📥 Cloning repository..."
git clone -b djoyce/vision-encoder-llava https://github.com/YOUR_USERNAME/gpt-oss-vision-encoder.git
cd gpt-oss-vision-encoder

# 3. Install Python dependencies
echo "🐍 Installing Python packages..."
pip install --no-cache-dir \
    datasets==4.0.0 \
    psutil \
    accelerate \
    wandb \
    tensorboard

# 4. Test environment
echo "🧪 Testing environment..."
python test_docker_local.py

# 5. Run quick training
echo "🎯 Starting quick training (1K samples for testing)..."
python -c "
import sys
sys.path.append('.')

print('🏃 Running 1K sample proof-of-concept...')

# Import your quick test
exec(open('train_mini_quick.py').read())
"

echo "✅ Setup complete! Ready for training."