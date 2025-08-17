# 🚀 RunPod GPU Training Guide for GPT-OSS Vision

Complete guide for training your multimodal model on RunPod with LLaVA datasets.

## 📊 Recommended Training Datasets

### 1. **LLaVA-Instruct-150K** (Recommended Start)
The same dataset used by LLaVA for visual instruction tuning.

```python
from datasets import load_dataset

# Load from HuggingFace
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
```

**What it contains:**
- 150K GPT-4 generated visual conversations
- Three types: conversations, detailed descriptions, complex reasoning
- Based on COCO images
- Perfect for instruction following

### 2. **LLaVA v1.5 Full Mix (665K)**
For more comprehensive training:

```bash
# Download annotation file
wget https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_v1_5_mix665k.json

# You'll also need images from:
# - COCO train2017
# - GQA images
# - OCR-VQA
# - TextVQA
# - VisualGenome
```

### 3. **Stage-by-Stage Training Data**

#### Stage 1: Pretrain (558K)
```python
# Feature alignment stage
# LAION-CC-SBU dataset subset
dataset = load_dataset("liuhaotian/LLaVA-Pretrain", split="train")
```

#### Stage 2: Instruct (150K + 515K)
```python
# Visual instruction tuning
instruct_data = load_dataset("liuhaotian/LLaVA-Instruct-150K")
# Plus academic VQA datasets
```

## 💻 RunPod Setup Guide

### Step 1: Create RunPod Account
1. Sign up at [runpod.io](https://runpod.io)
2. Verify email
3. Enable 2FA (recommended)
4. Add credits ($10-20 to start)

### Step 2: Choose GPU Configuration

For LLaVA training, recommended specs:

| Training Stage | GPU | VRAM | Price/hr | Time Estimate |
|---------------|-----|------|----------|---------------|
| Testing | RTX 4090 | 24GB | $0.34 | 1-2 hours |
| Stage 1 | RTX 6000 Ada | 48GB | $0.79 | 4-6 hours |
| Stage 2 | A100 40GB | 40GB | $1.14 | 6-8 hours |
| Full Training | A100 80GB | 80GB | $1.99 | 12-24 hours |

### Step 3: Deploy Your Pod

1. Go to **Pods** → **Deploy**
2. Select GPU (start with RTX 4090 for testing)
3. Choose template: **PyTorch 2.2.0 + CUDA 12.1**
4. Configure:
   ```
   Container Disk: 50 GB
   Volume Disk: 100 GB (for datasets)
   ```
5. Click **Deploy On-Demand**

### Step 4: Connect and Setup Environment

```bash
# Connect via SSH or Jupyter
ssh root@[your-pod-ip] -p [port]

# Or use Web Terminal
# Click "Connect" → "Connect to Web Terminal"
```

## 📦 Installation Script for RunPod

Create `setup_training.sh`:

```bash
#!/bin/bash

# Update system
apt-get update && apt-get install -y git wget

# Clone your repo
git clone https://github.com/yourusername/gpt-oss-vision-encoder.git
cd gpt-oss-vision-encoder

# Create virtual environment
python -m venv train_env
source train_env/bin/activate

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pillow numpy accelerate wandb

# Download CLIP model (cache it)
python -c "from transformers import CLIPVisionModel; CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14')"

echo "✅ Setup complete!"
```

## 🎯 Training Script for RunPod

Create `train_on_llava.py`:

```python
#!/usr/bin/env python3
"""
Full training script for RunPod with LLaVA datasets
"""

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import CLIPVisionModel, CLIPImageProcessor
from accelerate import Accelerator
import wandb

# Initialize accelerator for multi-GPU if available
accelerator = Accelerator()

# Configuration
config = {
    'stage1_epochs': 1,
    'stage2_epochs': 3,
    'batch_size': 4 if torch.cuda.device_count() == 1 else 8,
    'learning_rate': 2e-5,
    'warmup_steps': 500,
    'logging': True
}

# Initialize wandb for tracking
if config['logging']:
    wandb.init(project="gpt-oss-vision", config=config)

print(f"🚀 Training on {torch.cuda.device_count()} GPU(s)")

# Load model (your existing architecture)
from gpt_oss.torch.model import Transformer, ModelConfig
model_config = ModelConfig(
    num_hidden_layers=12,  # Bigger for real training
    hidden_size=768,
    vocab_size=50258,
    num_attention_heads=12,
    num_key_value_heads=12,
    intermediate_size=3072,
)

model = Transformer(model_config, device='cuda')

# Load CLIP (large version for better quality)
vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

# ============================================
# STAGE 1: Pretrain on LAION-CC-SBU
# ============================================

print("="*60)
print("STAGE 1: Vision-Language Alignment")
print("="*60)

# Load pretrain dataset
pretrain_data = load_dataset("liuhaotian/LLaVA-Pretrain", split="train")

# Your Stage 1 training loop here
# ... (use your existing train_stage1 function)

torch.save(model.mm_projector.state_dict(), "stage1_projector.pt")
print("✅ Stage 1 complete!")

# ============================================
# STAGE 2: Instruct Tuning
# ============================================

print("="*60)
print("STAGE 2: Visual Instruction Tuning")
print("="*60)

# Load instruction dataset
instruct_data = load_dataset("liuhaotian/LLaVA-Instruct-150K")

# Process into conversation format
def process_llava_conversations(item):
    """Convert LLaVA format to your training format"""
    conversations = []
    for conv in item['conversations']:
        if conv['from'] == 'human':
            question = conv['value'].replace('<image>', DEFAULT_IMAGE_TOKEN)
        else:
            answer = conv['value']
            conversations.append({
                'question': question,
                'answer': answer,
                'image': item['image']
            })
    return conversations

# Your Stage 2 training loop here
# ... (use your existing train_stage2 function)

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'projector_state_dict': model.mm_projector.state_dict(),
    'config': model_config
}, "llava_trained_model.pt")

print("🎉 Training complete!")
```

## ⚡ Optimizations for RunPod

### 1. Use Mixed Precision Training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 2. Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```

### 3. Multi-GPU Training
```python
# RunPod supports multi-GPU pods
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
```

### 4. Use Persistent Storage
```bash
# Save datasets to volume (persists between sessions)
cd /workspace
wget [dataset_urls]
```

## 📊 Monitoring Training

### RunPod Dashboard
- Monitor GPU usage in real-time
- Check VRAM consumption
- View logs

### Weights & Biases Integration
```python
wandb.log({
    'loss': loss.item(),
    'learning_rate': lr,
    'epoch': epoch
})
```

### Save Checkpoints Regularly
```python
if step % 1000 == 0:
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, f'/workspace/checkpoint_{step}.pt')
```

## 💰 Cost Optimization Tips

1. **Start Small**: Test with RTX 4090 ($0.34/hr) before scaling up
2. **Use Spot Instances**: Up to 50% cheaper but can be interrupted
3. **Stop Pods When Not Using**: Avoid idle charges
4. **Download Data Once**: Store in persistent volume
5. **Use Gradient Accumulation**: Simulate larger batches on smaller GPUs

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## 🚀 Quick Start Commands

```bash
# 1. Deploy Pod (via RunPod UI)

# 2. Connect and setup
ssh root@[pod-ip] -p [port]
wget https://raw.githubusercontent.com/[your-repo]/setup_training.sh
bash setup_training.sh

# 3. Download LLaVA data
python -c "from datasets import load_dataset; load_dataset('liuhaotian/LLaVA-Instruct-150K')"

# 4. Start training
python train_on_llava.py

# 5. Monitor with tensorboard
tensorboard --logdir=./logs --bind_all
```

## 📈 Expected Results

| Dataset Size | Training Time (A100) | Expected Performance |
|-------------|---------------------|---------------------|
| 10K samples | 1-2 hours | Basic object recognition |
| 50K samples | 4-6 hours | Good descriptions |
| 150K samples | 12-16 hours | Detailed conversations |
| 665K samples | 24-48 hours | Near LLaVA quality |

## 🔧 Troubleshooting

### CUDA Out of Memory
```python
# Reduce batch size
config['batch_size'] = 2

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use smaller model
config['hidden_size'] = 512
```

### Slow Training
```python
# Use larger GPU (A100 80GB)
# Enable mixed precision
# Increase batch size
# Use data parallelism
```

### Connection Issues
```bash
# Use RunPod's Web Terminal as backup
# Or use Jupyter notebook interface
```

## 📚 Additional Resources

- [LLaVA Paper](https://arxiv.org/abs/2304.08485)
- [RunPod Documentation](https://docs.runpod.io)
- [HuggingFace Datasets](https://huggingface.co/datasets)
- [Your Phase 4 Implementation](./PHASE4_COMPLETION_SUMMARY.md)

---

**Ready to train!** Start with a small subset on an RTX 4090 to verify everything works, then scale up to A100 for full training. The same architecture that works with shape recognition will work with LLaVA's rich visual conversations!