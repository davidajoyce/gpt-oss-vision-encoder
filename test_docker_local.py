#!/usr/bin/env python3
"""
Local Docker test script - validates all dependencies and imports work correctly
This will verify everything is ready before deploying to RunPod
"""

import sys
import os
import time
from datetime import datetime

print("="*60)
print("🐳 Docker Environment Test - Local Validation")
print("="*60)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Track what works and what doesn't
test_results = {}

def test_step(name, test_func):
    """Run a test step and track results"""
    try:
        start_time = time.time()
        result = test_func()
        elapsed = time.time() - start_time
        print(f"✅ {name} ({elapsed:.2f}s)")
        test_results[name] = {"status": "PASS", "time": elapsed, "result": result}
        return result
    except Exception as e:
        print(f"❌ {name} - Error: {e}")
        test_results[name] = {"status": "FAIL", "error": str(e)}
        return None

# ============================================
# 1. BASIC PYTHON ENVIRONMENT
# ============================================

print("\n📦 Testing Python Environment...")

def test_python():
    import platform
    return f"Python {platform.python_version()}"

test_step("Python version", test_python)

# ============================================
# 2. PYTORCH AND CUDA
# ============================================

print("\n🔥 Testing PyTorch and CUDA...")

def test_torch():
    import torch
    return f"PyTorch {torch.__version__}"

def test_cuda():
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"CUDA available - {gpu_name} ({gpu_memory:.1f}GB)"
    else:
        return "CUDA not available (CPU only)"

test_step("PyTorch installation", test_torch)
test_step("CUDA availability", test_cuda)

# ============================================
# 3. TRANSFORMERS AND MODELS
# ============================================

print("\n🤖 Testing Transformers and Models...")

def test_transformers():
    import transformers
    return f"Transformers {transformers.__version__}"

def test_clip_import():
    from transformers import CLIPVisionModel, CLIPImageProcessor
    return "CLIP imports successful"

def test_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    return f"GPT-2 tokenizer loaded ({len(tokenizer)} tokens)"

test_step("Transformers library", test_transformers)
test_step("CLIP imports", test_clip_import)
test_step("Tokenizer loading", test_tokenizer)

# ============================================
# 4. GPT-OSS COMPONENTS
# ============================================

print("\n🧠 Testing GPT-OSS Components...")

def test_gpt_oss_imports():
    # Add current directory to path
    sys.path.insert(0, '/workspace/gpt-oss-vision' if os.path.exists('/workspace/gpt-oss-vision') else '.')
    
    from gpt_oss.torch.model import Transformer, ModelConfig
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    return "GPT-OSS imports successful"

def test_model_creation():
    import torch
    sys.path.insert(0, '/workspace/gpt-oss-vision' if os.path.exists('/workspace/gpt-oss-vision') else '.')
    
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
    param_count = sum(p.numel() for p in model.parameters())
    return f"Model created with {param_count:,} parameters on {device}"

def test_multimodal_generator():
    sys.path.insert(0, '/workspace/gpt-oss-vision' if os.path.exists('/workspace/gpt-oss-vision') else '.')
    
    from gpt_oss.generate_multimodal import MultimodalTextGenerator
    return "MultimodalTextGenerator import successful"

test_step("GPT-OSS imports", test_gpt_oss_imports)
test_step("Model creation", test_model_creation) 
test_step("Multimodal generator", test_multimodal_generator)

# ============================================
# 5. IMAGE PROCESSING
# ============================================

print("\n🖼️ Testing Image Processing...")

def test_pil():
    from PIL import Image, ImageDraw
    # Create test image
    img = Image.new('RGB', (224, 224), 'red')
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 174, 174], fill='blue')
    return f"PIL working - created {img.size} image"

def test_numpy():
    import numpy as np
    arr = np.random.randn(224, 224, 3)
    return f"NumPy working - array shape {arr.shape}"

test_step("PIL image processing", test_pil)
test_step("NumPy arrays", test_numpy)

# ============================================
# 6. DATASETS AND DATA LOADING
# ============================================

print("\n📚 Testing Datasets...")

def test_datasets():
    from datasets import Dataset
    # Create dummy dataset
    data = [{"text": f"sample {i}", "label": i} for i in range(10)]
    dataset = Dataset.from_list(data)
    return f"Datasets working - created dataset with {len(dataset)} samples"

def test_torch_dataloader():
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # Create dummy data
    x = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=8)
    
    batch_count = len(dataloader)
    return f"DataLoader working - {batch_count} batches"

test_step("HuggingFace datasets", test_datasets)
test_step("PyTorch DataLoader", test_torch_dataloader)

# ============================================
# 7. FULL INTEGRATION TEST
# ============================================

print("\n🔄 Testing Full Integration...")

def test_end_to_end():
    """Complete end-to-end test of the pipeline"""
    sys.path.insert(0, '/workspace/gpt-oss-vision' if os.path.exists('/workspace/gpt-oss-vision') else '.')
    
    import torch
    from PIL import Image
    from transformers import CLIPVisionModel, CLIPImageProcessor
    from gpt_oss.torch.model import Transformer, ModelConfig
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN
    
    # 1. Create tiny model
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=32,
        vocab_size=100,
        num_attention_heads=2,
        num_key_value_heads=2,
        intermediate_size=64,
        initial_context_length=128,
        head_dim=16,
        sliding_window=32
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(config, device=device)
    
    # 2. Create simple tokenizer
    class SimpleTokenizer:
        def __init__(self):
            self.vocab = {"<pad>": 0, "<unk>": 1, "test": 2}
            self.unk_token_id = 1
        
        def encode(self, text, return_tensors=None):
            ids = [2, 2, 2]  # Simple test tokens
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
    
    tokenizer = SimpleTokenizer()
    
    # 3. Initialize image tokenizer
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    model.set_image_token_id(image_token_id)
    
    # 4. Test image processing
    test_image = Image.new('RGB', (224, 224), 'blue')
    
    # 5. Test generation
    from gpt_oss.generate_multimodal import MultimodalTextGenerator
    generator = MultimodalTextGenerator(model, tokenizer)
    
    response = generator.generate_response(
        prompt=f"What is {DEFAULT_IMAGE_TOKEN}?",
        image=test_image,
        max_tokens=3,
        temperature=0.1
    )
    
    return f"End-to-end test successful - generated: '{response}'"

test_step("End-to-end pipeline", test_end_to_end)

# ============================================
# 8. PERFORMANCE TEST
# ============================================

print("\n⚡ Testing Performance...")

def test_memory_usage():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return f"Memory usage: {memory_mb:.1f} MB"

def test_tensor_operations():
    import torch
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test tensor operations
    x = torch.randn(1000, 1000, device=device)
    y = torch.randn(1000, 1000, device=device)
    
    start = time.time()
    z = torch.matmul(x, y)
    elapsed = time.time() - start
    
    return f"Matrix multiply (1000x1000) on {device}: {elapsed:.3f}s"

test_step("Memory usage", test_memory_usage)
test_step("Tensor operations", test_tensor_operations)

# ============================================
# 9. RESULTS SUMMARY
# ============================================

print("\n" + "="*60)
print("📊 TEST RESULTS SUMMARY")
print("="*60)

passed = sum(1 for r in test_results.values() if r["status"] == "PASS")
failed = sum(1 for r in test_results.values() if r["status"] == "FAIL")
total = len(test_results)

print(f"\n✅ Passed: {passed}/{total}")
print(f"❌ Failed: {failed}/{total}")

if failed > 0:
    print(f"\n🔴 FAILED TESTS:")
    for name, result in test_results.items():
        if result["status"] == "FAIL":
            print(f"  - {name}: {result['error']}")

print(f"\n🕐 Total test time: {sum(r.get('time', 0) for r in test_results.values()):.2f}s")

# Environment info
print(f"\n🔧 ENVIRONMENT INFO:")
print(f"  - Working directory: {os.getcwd()}")
print(f"  - Python path: {sys.executable}")
print(f"  - Available CPUs: {os.cpu_count()}")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  - GPU: {torch.cuda.get_device_name(0)}")
        print(f"  - CUDA version: {torch.version.cuda}")
    else:
        print(f"  - GPU: None (CPU only)")
except:
    print(f"  - GPU: Could not check (torch import failed)")

# Docker specific checks
if os.path.exists('/.dockerenv'):
    print(f"  - Running in Docker: ✅")
else:
    print(f"  - Running in Docker: ❌ (local environment)")

print(f"\n🎯 DOCKER READINESS:")
if failed == 0:
    print("✅ Ready for RunPod deployment!")
    print("✅ All dependencies working correctly")
    print("✅ GPT-OSS components functional")
else:
    print("❌ Fix failed tests before deploying")

print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)