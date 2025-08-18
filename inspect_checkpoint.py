#!/usr/bin/env python3
"""
Inspect what's inside a checkpoint file
"""

import sys
import torch
import os

def inspect_checkpoint(path):
    """Inspect checkpoint contents without loading everything into memory"""
    print(f"📊 Inspecting: {path}")
    print(f"📦 File size: {os.path.getsize(path) / 1e9:.2f}GB\n")
    
    try:
        # Load with map_location to avoid GPU memory
        print("Loading checkpoint keys (not values)...")
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        
        print(f"✅ Checkpoint contains {len(checkpoint)} keys:\n")
        
        # Show all keys and their sizes
        for key in checkpoint.keys():
            value = checkpoint[key]
            
            if isinstance(value, torch.Tensor):
                size_mb = value.numel() * value.element_size() / 1e6
                print(f"  {key}: Tensor {value.shape} ({size_mb:.1f}MB)")
            elif isinstance(value, dict):
                # Count tensors in dict
                num_tensors = sum(1 for v in value.values() if isinstance(v, torch.Tensor))
                total_size = 0
                for v in value.values():
                    if isinstance(v, torch.Tensor):
                        total_size += v.numel() * v.element_size() / 1e6
                print(f"  {key}: Dict with {len(value)} items, {num_tensors} tensors ({total_size:.1f}MB)")
            else:
                print(f"  {key}: {type(value).__name__}")
        
        # Check for specific issues
        print("\n🔍 Checking for issues...")
        
        # Check if optimizer state is huge
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            if 'state' in opt_state:
                opt_size = 0
                for param_state in opt_state['state'].values():
                    if isinstance(param_state, dict):
                        for k, v in param_state.items():
                            if isinstance(v, torch.Tensor):
                                opt_size += v.numel() * v.element_size() / 1e9
                print(f"  Optimizer state size: {opt_size:.2f}GB")
                if opt_size > 1:
                    print(f"  ⚠️ WARNING: Optimizer state is very large!")
        
        # Check model size
        if 'model_state_dict' in checkpoint:
            model_size = sum(
                v.numel() * v.element_size() / 1e6 
                for v in checkpoint['model_state_dict'].values() 
                if isinstance(v, torch.Tensor)
            )
            print(f"  Model weights size: {model_size:.1f}MB")
            
        # Check for duplicates or unexpected keys
        unexpected_keys = [k for k in checkpoint.keys() if k not in [
            'model_state_dict', 'projector_state_dict', 'optimizer_state_dict',
            'scheduler_state_dict', 'config', 'model_config', 'step', 'epoch',
            'loss', 'total_time', 'final_loss', 'best_loss', 'total_steps',
            'training_complete', 'used_real_data', 'model_size_mb', 
            'gpu_peak_memory', 'gpu_memory_peak', 'timestamp'
        ]]
        
        if unexpected_keys:
            print(f"\n⚠️ Unexpected keys found:")
            for key in unexpected_keys:
                print(f"    - {key}")
                
    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        return
    
    print("\n💡 Recommendations:")
    file_size_gb = os.path.getsize(path) / 1e9
    if file_size_gb > 5:
        print("  This file is WAY too large for a vision-language model!")
        print("  Normal size should be <500MB")
        print("  Consider extracting just the model weights")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)
    
    inspect_checkpoint(sys.argv[1])