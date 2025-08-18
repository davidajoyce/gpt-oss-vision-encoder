#!/usr/bin/env python3
"""
Debug what's wrong with the checkpoint
"""

import torch
import sys

def debug_checkpoint(path):
    print(f"🔍 Deep inspection of: {path}\n")
    
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Check model state dict
    model_state = checkpoint['model_state_dict']
    print(f"📊 Model state dict has {len(model_state)} keys\n")
    
    # Group by parameter type
    param_groups = {}
    total_params = 0
    
    for key, tensor in model_state.items():
        # Get base name without numbers
        base_name = key.split('.')[0] if '.' in key else key
        
        if base_name not in param_groups:
            param_groups[base_name] = []
        
        param_size = tensor.numel()
        total_params += param_size
        param_groups[base_name].append((key, tensor.shape, param_size))
    
    print("🔢 Parameter groups:")
    for group, params in param_groups.items():
        group_size = sum(p[2] for p in params)
        print(f"\n  {group}: {len(params)} tensors, {group_size/1e6:.1f}M params")
        
        # Show first few
        for i, (name, shape, size) in enumerate(params[:3]):
            print(f"    - {name}: {shape} ({size/1e6:.2f}M params)")
        if len(params) > 3:
            print(f"    ... and {len(params)-3} more")
    
    print(f"\n📈 Total parameters: {total_params/1e6:.1f}M")
    print(f"📦 Expected size: {total_params * 4 / 1e9:.2f}GB (float32)")
    print(f"❌ Actual size: {sum(t.numel() * t.element_size() for t in model_state.values()) / 1e9:.2f}GB")
    
    # Check for duplicate keys or weird patterns
    print("\n🔍 Checking for anomalies...")
    
    # Check tensor data types
    dtypes = {}
    for key, tensor in model_state.items():
        dtype = str(tensor.dtype)
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += 1
    
    print(f"\n📊 Data types:")
    for dtype, count in dtypes.items():
        print(f"  {dtype}: {count} tensors")
    
    # Check for unexpectedly large tensors
    print("\n⚠️ Largest tensors:")
    sorted_tensors = sorted(
        [(k, v.numel() * v.element_size() / 1e9) for k, v in model_state.items()],
        key=lambda x: x[1],
        reverse=True
    )
    for name, size_gb in sorted_tensors[:5]:
        tensor = model_state[name]
        print(f"  {name}: {tensor.shape}, {size_gb:.2f}GB, dtype={tensor.dtype}")
    
    # Check if tensors are actually float64 or something
    first_tensor = next(iter(model_state.values()))
    actual_bytes_per_element = first_tensor.element_size()
    print(f"\n💾 Bytes per element: {actual_bytes_per_element}")
    if actual_bytes_per_element > 4:
        print(f"  ⚠️ WARNING: Using {actual_bytes_per_element} bytes per element instead of 4!")
        print(f"  This would explain the size: {total_params/1e6:.1f}M × {actual_bytes_per_element} = {total_params * actual_bytes_per_element / 1e9:.1f}GB")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_checkpoint.py <checkpoint>")
        sys.exit(1)
    
    debug_checkpoint(sys.argv[1])