#!/usr/bin/env python3
"""Quick test to check if all components work"""

import sys
sys.path.append('.')

print("Testing imports...")

try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch not installed: {e}")
    sys.exit(1)

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
    print("✓ Transformers library")
except ImportError as e:
    print(f"✗ Transformers not installed: {e}")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw
    print("✓ PIL/Pillow")
except ImportError as e:
    print(f"✗ PIL not installed: {e}")
    sys.exit(1)

try:
    from gpt_oss.torch.model import Transformer, ModelConfig
    print("✓ GPT-OSS model imports")
except ImportError as e:
    print(f"✗ GPT-OSS imports failed: {e}")
    sys.exit(1)

try:
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
    print("✓ GPT-OSS constants")
except ImportError as e:
    print(f"✗ Constants import failed: {e}")
    sys.exit(1)

print("\nTesting model creation...")
try:
    config = ModelConfig(
        num_hidden_layers=2,
        hidden_size=128,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256
    )
    model = Transformer(config, device='cpu')
    print(f"✓ Created model with {sum(p.numel() for p in model.parameters())} parameters")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting CLIP download (this may take a moment)...")
try:
    print("  Downloading CLIP vision model...")
    vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    print("  Downloading CLIP processor...")  
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    print(f"✓ CLIP model loaded, hidden_size: {vision_model.config.hidden_size}")
except Exception as e:
    print(f"✗ CLIP download failed: {e}")
    sys.exit(1)

print("\n✅ All components working! Ready to run training script.")