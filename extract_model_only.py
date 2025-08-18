#!/usr/bin/env python3
"""
Extract just the model weights from a bloated checkpoint
"""

import sys
import torch
import os

def extract_model(input_path, output_path="model_extracted.pt"):
    """Extract only essential model weights"""
    print(f"📊 Extracting from: {input_path}")
    print(f"📦 Original size: {os.path.getsize(input_path) / 1e9:.2f}GB")
    
    try:
        print("\n⏳ Loading checkpoint (this may take a while for 42GB)...")
        checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
        
        print("✅ Checkpoint loaded, extracting essentials...")
        
        # Create minimal checkpoint with just what's needed
        minimal = {
            'model_state_dict': checkpoint.get('model_state_dict', {}),
            'projector_state_dict': checkpoint.get('projector_state_dict', {}),
            'model_config': checkpoint.get('model_config'),
            'training_complete': True,
            'final_loss': checkpoint.get('final_loss', checkpoint.get('loss', 0)),
            'total_steps': checkpoint.get('total_steps', checkpoint.get('step', 0)),
            'epoch': checkpoint.get('epoch', 3)
        }
        
        # Calculate new size
        model_size = sum(
            v.numel() * v.element_size() / 1e6 
            for v in minimal['model_state_dict'].values() 
            if isinstance(v, torch.Tensor)
        )
        projector_size = sum(
            v.numel() * v.element_size() / 1e6 
            for v in minimal['projector_state_dict'].values() 
            if isinstance(v, torch.Tensor)
        )
        
        print(f"\n📊 Extracted sizes:")
        print(f"  Model: {model_size:.1f}MB")
        print(f"  Projector: {projector_size:.1f}MB")
        print(f"  Total: {(model_size + projector_size):.1f}MB")
        
        print(f"\n💾 Saving to: {output_path}")
        torch.save(minimal, output_path)
        
        new_size = os.path.getsize(output_path) / 1e6
        print(f"✅ Saved! New size: {new_size:.1f}MB")
        print(f"🎉 Reduced from {os.path.getsize(input_path)/1e9:.1f}GB to {new_size:.1f}MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_model_only.py <input_checkpoint> [output_path]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "model_extracted.pt"
    
    result = extract_model(input_path, output_path)
    if result:
        print(f"\n🧪 You can now test with:")
        print(f"python quick_image_test.py {result}")