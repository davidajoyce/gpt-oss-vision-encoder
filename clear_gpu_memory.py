#!/usr/bin/env python3
"""
GPU Memory Cleanup Script
Clears all GPU memory and processes before training
"""

import torch
import gc
import os
import subprocess

print("🔧 GPU Memory Cleanup Script")
print("=" * 40)

# Clear Python GPU memory
if torch.cuda.is_available():
    print("📊 Before cleanup:")
    print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"  GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    # Aggressive cleanup
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n✅ After cleanup:")
    print(f"  GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"  GPU Memory Reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    available = total_memory - torch.cuda.memory_allocated()/1e9
    print(f"  Available Memory: {available:.1f}GB / {total_memory:.1f}GB")
    
    if available < 5:
        print(f"\n⚠️ WARNING: Low available memory ({available:.1f}GB)")
        print("🔧 Consider killing other Python processes:")
        print("   pkill -f python")
        print("   nvidia-smi")
    else:
        print(f"\n✅ Good available memory ({available:.1f}GB)")
        print("🚀 Ready for training!")

else:
    print("❌ No CUDA GPU available")

print("\n🔍 Current GPU processes:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.split('\n')
        # Find processes section
        for i, line in enumerate(lines):
            if 'Processes:' in line:
                for process_line in lines[i:i+10]:
                    if 'python' in process_line.lower():
                        print(f"  {process_line.strip()}")
                break
    else:
        print("  nvidia-smi not available")
except:
    print("  Could not check GPU processes")

print("\n🏁 Cleanup complete!")