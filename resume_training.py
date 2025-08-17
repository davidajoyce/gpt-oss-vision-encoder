#!/usr/bin/env python3
"""
Resume Training Script
Automatically finds the latest checkpoint and resumes training
"""

import os
import glob
import torch
import re

print("🔄 LLaVA Training Recovery Script")
print("=" * 40)

# Find all checkpoint files
checkpoint_pattern = "llava_150k_ultimate_checkpoint_step_*.pt"
checkpoint_files = glob.glob(checkpoint_pattern)

if not checkpoint_files:
    print("❌ No checkpoint files found!")
    print(f"   Looking for: {checkpoint_pattern}")
    print("🔧 Starting fresh training...")
    os.system("python train_llava_150k_ultimate.py")
    exit()

# Find the latest checkpoint by step number
latest_checkpoint = None
latest_step = 0

for checkpoint_file in checkpoint_files:
    # Extract step number from filename
    match = re.search(r'step_(\d+)\.pt', checkpoint_file)
    if match:
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_checkpoint = checkpoint_file

if latest_checkpoint:
    print(f"✅ Found latest checkpoint: {latest_checkpoint}")
    print(f"📊 Last completed step: {latest_step}")
    
    # Load checkpoint to check status
    try:
        checkpoint = torch.load(latest_checkpoint, map_location='cpu')
        
        print(f"🔍 Checkpoint Info:")
        print(f"  - Step: {checkpoint.get('step', 'unknown')}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  - Loss: {checkpoint.get('loss', 'unknown'):.4f}")
        print(f"  - Used real data: {checkpoint.get('used_real_data', 'unknown')}")
        
        if 'gpu_memory_peak' in checkpoint:
            print(f"  - Peak GPU memory: {checkpoint['gpu_memory_peak']:.1f}GB")
        
        print(f"\n🚀 Resuming training from step {latest_step}...")
        
        # TODO: Implement resume logic in main training script
        # For now, just restart fresh
        print("⚠️ Resume logic not implemented yet, starting fresh...")
        
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        print("🔧 Starting fresh training...")
    
else:
    print("❌ Could not parse checkpoint filenames")
    print("🔧 Starting fresh training...")

# Start training
print("\n" + "=" * 40)
os.system("python train_llava_150k_ultimate.py")