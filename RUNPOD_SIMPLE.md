# ⚡ Super Simple RunPod Setup

Get your vision model trained in 1-2 hours for ~$0.70

## 🎯 RTX 4090 Training Timeline

| What | Time | Cost |
|------|------|------|
| **10K samples** | 1-2 hours | $0.70 |
| **Proof it works** | 30 mins | $0.20 |

## 🐳 Docker Method (Easiest)

### Step 1: Deploy RunPod
1. Go to [runpod.io](https://runpod.io) → Sign up
2. Add $5 credits
3. **Deploy** → **RTX 4090** → **PyTorch 2.2** template
4. Set:
   ```
   Container Disk: 40GB
   Volume Disk: 20GB
   ```
5. **Deploy On-Demand**

### Step 2: One Command Setup
Connect to your pod and run:

```bash
# Copy this entire block and paste into RunPod terminal
git clone https://github.com/YOUR_USERNAME/gpt-oss-vision-encoder.git && \
cd gpt-oss-vision-encoder && \
docker build -t gpt-oss-vision . && \
docker run --gpus all \
  -v $(pwd)/checkpoints:/workspace/checkpoints \
  gpt-oss-vision \
  python train_quick_10k.py
```

**That's it!** Training starts automatically.

## 📊 What Happens

1. **Downloads** LLaVA dataset (10K samples)
2. **Trains** your model for 2 epochs (~1.5 hours)
3. **Saves** checkpoints every 2500 steps
4. **Tests** generation at the end

## 📈 Real-Time Monitoring

Watch the training progress:
```
Epoch 1/2
  Batch 100/2500 | Loss: 2.4532 | Batch/s: 1.2 | ETA: 45min
  Batch 200/2500 | Loss: 1.8234 | Batch/s: 1.3 | ETA: 42min
  💾 Saved checkpoint: checkpoint_step_2500.pt
```

## 💰 Cost Breakdown
- **RTX 4090**: $0.34/hour
- **10K training**: 1.5 hours = $0.51
- **Data download**: 15 mins = $0.09
- **Total**: ~$0.60

## 🎉 Expected Results

After training:
```python
Input:  "What do you see in <image>?" + [red square image]
Output: "I see a red geometric shape" (instead of random tokens)
```

## 🔧 Even Simpler: Use Pre-built Image

If you want to skip building Docker:

```bash
# Use pre-built image (when available)
docker pull ghcr.io/your-username/gpt-oss-vision:latest
docker run --gpus all ghcr.io/your-username/gpt-oss-vision:latest
```

## 🚨 If Something Goes Wrong

### Common Issues:
1. **Docker not found**: RunPod should have it pre-installed
2. **GPU not detected**: Make sure you selected RTX 4090 pod
3. **Out of memory**: Training script is optimized for 24GB VRAM

### Quick Fixes:
```bash
# Check GPU
nvidia-smi

# Free up memory
docker system prune -f

# Restart if needed
docker restart [container-id]
```

## ⏱️ Timeline

| Time | What's Happening |
|------|------------------|
| 0-5 mins | Pod deployment |
| 5-10 mins | Code download & Docker build |
| 10-20 mins | Dataset download |
| 20-120 mins | Training (main time) |
| 120+ mins | Model ready! |

## 📁 What You Get

After training completes:
```
checkpoints/
├── checkpoint_step_2500.pt    # Mid-training
├── checkpoint_step_5000.pt    # Mid-training  
├── checkpoint_step_7500.pt    # Mid-training
└── llava_10k_rtx4090_final.pt # Final model
```

## 🎮 Test Your Model

After training, test it:
```bash
python test_trained_model.py \
  --model checkpoints/llava_10k_rtx4090_final.pt \
  --image test_image.jpg \
  --prompt "What do you see in <image>?"
```

---

**Total time investment**: 5 minutes setup + 1.5 hours automated training = Your own vision-language model for less than $1! 🚀