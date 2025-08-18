#!/usr/bin/env python3
"""
Fix and extract model with proper dtype
"""

import torch
import sys
import os

def fix_checkpoint(input_path, output_path="model_fixed.pt"):
    print(f"🔧 Fixing checkpoint: {input_path}")
    print(f"📦 Original size: {os.path.getsize(input_path) / 1e9:.2f}GB\n")
    
    print("⏳ Loading (this will take time for 42GB)...")
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    print("🔄 Converting model weights to float32...")
    
    # Fix model state dict - convert everything to float32
    fixed_model_state = {}
    for key, tensor in checkpoint['model_state_dict'].items():
        if tensor.dtype != torch.float32:
            print(f"  Converting {key} from {tensor.dtype} to float32")
            fixed_model_state[key] = tensor.float()
        else:
            fixed_model_state[key] = tensor
    
    # Fix projector too
    fixed_projector_state = {}
    for key, tensor in checkpoint['projector_state_dict'].items():
        if tensor.dtype != torch.float32:
            print(f"  Converting projector.{key} from {tensor.dtype} to float32")
            fixed_projector_state[key] = tensor.float()
        else:
            fixed_projector_state[key] = tensor
    
    # Create clean checkpoint
    clean_checkpoint = {
        'model_state_dict': fixed_model_state,
        'projector_state_dict': fixed_projector_state,
        'model_config': checkpoint['model_config'],
        'training_complete': True,
        'final_loss': checkpoint.get('final_loss', 0),
        'total_steps': checkpoint.get('total_steps', 0),
        'epoch': checkpoint.get('epoch', 3)
    }
    
    # Calculate expected size
    model_params = sum(t.numel() for t in fixed_model_state.values())
    projector_params = sum(t.numel() for t in fixed_projector_state.values())
    expected_size_mb = (model_params + projector_params) * 4 / 1e6
    
    print(f"\n📊 Fixed model stats:")
    print(f"  Model parameters: {model_params/1e6:.1f}M")
    print(f"  Projector parameters: {projector_params/1e6:.1f}M")
    print(f"  Expected size: {expected_size_mb:.1f}MB")
    
    print(f"\n💾 Saving to: {output_path}")
    torch.save(clean_checkpoint, output_path)
    
    actual_size_mb = os.path.getsize(output_path) / 1e6
    print(f"✅ Saved! Actual size: {actual_size_mb:.1f}MB")
    
    reduction = os.path.getsize(input_path) / os.path.getsize(output_path)
    print(f"🎉 Reduced by {reduction:.0f}x!")
    
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_and_extract.py <input_checkpoint> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model_fixed.pt"
    
    result = fix_checkpoint(input_path, output_path)
    if result:
        print(f"\n🧪 Test your fixed model:")
        print(f"python quick_image_test.py {result}")