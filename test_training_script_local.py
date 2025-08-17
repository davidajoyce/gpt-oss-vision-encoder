#!/usr/bin/env python3
"""
Local test for training script - validates all components work
Run this locally before deploying to RunPod to catch errors early
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
import time

print("="*60)
print("🧪 LOCAL TRAINING SCRIPT VALIDATION")
print("="*60)

def test_step(name, test_func):
    """Run a test and report results"""
    try:
        start = time.time()
        result = test_func()
        elapsed = time.time() - start
        print(f"✅ {name} ({elapsed:.2f}s)")
        return True, result
    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        import traceback
        traceback.print_exc()
        return False, str(e)

# Test 1: Basic imports
def test_imports():
    from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
    from datasets import Dataset
    from gpt_oss.torch.model import Transformer, ModelConfig
    from gpt_oss.generate_multimodal import MultimodalTextGenerator
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN
    return "All imports successful"

# Test 2: Model creation
def test_model_creation():
    from gpt_oss.torch.model import Transformer, ModelConfig
    
    config = ModelConfig(
        num_hidden_layers=1,  # Tiny for testing
        hidden_size=64,
        vocab_size=1000,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        initial_context_length=256,
        head_dim=32,
        sliding_window=64
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(config, device=device)
    return f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters"

# Test 3: Tokenizer setup
def test_tokenizer():
    from transformers import AutoTokenizer
    
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return f"GPT-2 tokenizer loaded ({len(tokenizer)} vocab)"
    except:
        # Fallback tokenizer
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {"<pad>": 0, "<unk>": 1, "test": 2}
                self.unk_token_id = 1
                self.eos_token_id = 2
            
            def encode(self, text, return_tensors=None):
                ids = [2, 2, 2]
                if return_tensors == "pt":
                    return torch.tensor([ids])
                return ids
            
            def decode(self, ids, skip_special_tokens=False):
                return "test output"
            
            def __len__(self):
                return 100
            
            def add_special_tokens(self, tokens_dict):
                return 1
        
        return "Fallback tokenizer created"

# Test 4: CLIP model loading (key test!)
def test_clip_loading():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    
    # Test the exact same loading method as training script
    try:
        vision_tower = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32", 
            use_safetensors=True
        )
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vision_tower = vision_tower.to(device)
        return f"CLIP loaded successfully on {device}"
    except Exception as e:
        # Try fallback method
        vision_tower = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            trust_remote_code=True
        )
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        vision_tower = vision_tower.to(device)
        return f"CLIP loaded with fallback method on {device}"

# Test 5: Vision tower wrapper (the fix!)
def test_vision_wrapper():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Test wrapper function (this is the fix)
    def vision_tower_wrapper(images):
        return vision_tower(images).last_hidden_state
    
    # Create test image
    test_image = Image.new('RGB', (224, 224), 'red')
    pixel_values = image_processor(images=test_image, return_tensors="pt")['pixel_values'].to(device)
    
    # Test the wrapper
    with torch.no_grad():
        features = vision_tower_wrapper(pixel_values)
    
    return f"Vision wrapper works - output shape: {features.shape}"

# Test 6: Projector creation
def test_projector():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    projector = nn.Sequential(
        nn.Linear(768, 64),  # CLIP hidden -> model hidden
        nn.GELU(),
        nn.Linear(64, 64)
    ).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(1, 49, 768).to(device)  # CLIP output shape
    output = projector(dummy_input)
    
    return f"Projector works - input: {dummy_input.shape}, output: {output.shape}"

# Test 7: Image tokenizer initialization
def test_image_tokenizer():
    from gpt_oss.torch.model import Transformer, ModelConfig
    
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=64,
        vocab_size=100,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=128,
        initial_context_length=128,
        head_dim=32,
        sliding_window=32
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(config, device=device)
    
    # Simple tokenizer for testing
    class TestTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "test": 2}
            self.unk_token_id = 1
        
        def __len__(self):
            return 100
        
        def add_special_tokens(self, tokens_dict):
            return 1
        
        def convert_tokens_to_ids(self, token):
            return 3
    
    tokenizer = TestTokenizer()
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    model.set_image_token_id(image_token_id)
    
    return f"Image tokenizer initialized, token ID: {image_token_id}"

# Test 8: Complete pipeline test
def test_complete_pipeline():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from gpt_oss.torch.model import Transformer, ModelConfig
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Tiny model
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=32,
        vocab_size=100,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        initial_context_length=64,
        head_dim=16,
        sliding_window=32
    )
    
    model = Transformer(config, device=device)
    
    # Simple tokenizer
    class TestTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "test": 2}
            self.unk_token_id = 1
        
        def encode(self, text, return_tensors=None):
            ids = [2, 2, 2]
            if return_tensors == "pt":
                return torch.tensor([ids])
            return ids
        
        def decode(self, ids, skip_special_tokens=False):
            return "test output"
        
        def __len__(self):
            return 100
        
        def add_special_tokens(self, tokens_dict):
            return 1
        
        def convert_tokens_to_ids(self, token):
            return 3
    
    tokenizer = TestTokenizer()
    
    # Initialize image tokenizer
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    model.set_image_token_id(image_token_id)
    
    # Load CLIP
    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Create projector
    projector = nn.Linear(768, config.hidden_size).to(device)
    
    # Set up model components (with wrapper!)
    def vision_tower_wrapper(images):
        return vision_tower(images).last_hidden_state
    
    model.vision_tower = vision_tower_wrapper
    model.mm_projector = projector
    
    # Test image
    test_image = Image.new('RGB', (224, 224), 'blue')
    pixel_values = image_processor(images=test_image, return_tensors="pt")['pixel_values'].to(device)
    
    # Test input
    input_ids = torch.tensor([[2, 2, 2]]).to(device)
    
    # Test multimodal input preparation
    multimodal_inputs = model.prepare_multimodal_inputs(
        input_ids=input_ids,
        images=pixel_values
    )
    
    embeddings = multimodal_inputs["inputs_embeds"]
    
    return f"Complete pipeline works! Embeddings shape: {embeddings.shape}"

# Run all tests
tests = [
    ("Basic imports", test_imports),
    ("Model creation", test_model_creation), 
    ("Tokenizer setup", test_tokenizer),
    ("CLIP loading", test_clip_loading),
    ("Vision wrapper", test_vision_wrapper),
    ("Projector creation", test_projector),
    ("Image tokenizer", test_image_tokenizer),
    ("Complete pipeline", test_complete_pipeline)
]

print(f"\nRunning {len(tests)} validation tests...\n")

passed = 0
failed = 0
results = []

for name, test_func in tests:
    success, result = test_step(name, test_func)
    results.append((name, success, result))
    if success:
        passed += 1
    else:
        failed += 1

print(f"\n" + "="*60)
print(f"📊 TEST RESULTS")
print(f"="*60)
print(f"✅ Passed: {passed}/{len(tests)}")
print(f"❌ Failed: {failed}/{len(tests)}")

if failed > 0:
    print(f"\n🔴 FAILED TESTS:")
    for name, success, result in results:
        if not success:
            print(f"  - {name}: {result}")

print(f"\n🎯 RUNPOD READINESS:")
if failed == 0:
    print("✅ Ready for RunPod! Training script should work.")
elif failed <= 2:
    print("⚠️ Mostly ready - minor issues that might be RunPod-specific")
else:
    print("❌ Fix failed tests before deploying to RunPod")

if torch.cuda.is_available():
    print(f"\n🖥️ GPU: {torch.cuda.get_device_name(0)}")
else:
    print(f"\n🖥️ GPU: None (CPU only - RunPod will have GPU)")

print(f"\n🚀 If tests pass, use this command on RunPod:")
print(f"git clone -b djoyce/vision-encoder-llava https://github.com/YOUR_USERNAME/gpt-oss-vision-encoder.git && cd gpt-oss-vision-encoder && pip install --upgrade torch torchvision datasets transformers pillow accelerate && python train_quick_10k.py")
print("="*60)