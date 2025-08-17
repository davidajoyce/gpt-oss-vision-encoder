# 🧪 Vision Encoder Testing Guide

This guide covers all tests implemented in the `djoyce/vision-encoder-llava` branch for GPT-OSS multimodal capabilities.

## 📋 Prerequisites

### 1. Set up Python environment
```bash
# Create and activate virtual environment
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install dependencies
pip install torch torchvision transformers pillow numpy
```

### 2. Verify setup
```bash
python test_setup.py
```

Expected output:
```
✓ PyTorch installed
✓ Transformers library
✓ PIL/Pillow
✓ GPT-OSS model imports
✓ GPT-OSS constants
✓ Created model with X parameters
✓ CLIP model loaded
✅ All components working!
```

## 🎯 Core Architecture Tests

### Phase 1: Vision Components (21 tests)
Tests basic vision encoding and feature extraction.

```bash
python test_phase1_basic.py
```
- Tests vision tower initialization
- Tests image processing pipeline
- Tests vision projector functionality
- Validates feature extraction shapes

### Phase 2: Training Infrastructure (20 tests)
Tests the two-stage training approach.

```bash
python tests/test_multimodal_trainer.py
```
- Stage 1: Vision-language alignment
- Stage 2: Instruction following
- Loss computation validation
- Training loop verification

### Phase 3: Inference Pipeline (30 tests)
Tests end-to-end multimodal generation.

```bash
python tests/test_generation_pipeline.py
```
- Multimodal input preparation
- Token generation with images
- Batch processing capabilities
- Error handling

## 🚀 Phase 4: Text Generation Tests

### 1. Image Token Infrastructure (21 tests)
```bash
python test_image_tokens.py
```
Tests:
- ✅ Image token constants (IMAGE_TOKEN_INDEX = -200)
- ✅ Tokenizer vocabulary extension
- ✅ Embedding resizing for new tokens
- ✅ Token validation and verification

### 2. Multimodal Input Preparation (18 tests)
```bash
python test_multimodal_inputs.py
```
Tests:
- ✅ Text-only backward compatibility
- ✅ Single image processing
- ✅ Multiple images in sequence
- ✅ Label alignment with IGNORE_INDEX
- ✅ Batch processing with padding

### 3. Text Generation Pipeline (14 tests)
```bash
python test_text_generation.py
```
Tests:
- ✅ Autoregressive generation
- ✅ Temperature sampling
- ✅ Greedy decoding
- ✅ Image context integration
- ✅ Generation with various parameters

## 🎮 Interactive Demos

### 1. Phase 4 Interactive Demo
```bash
python demo_phase4_interactive.py
```

This demo:
- Creates 9 test images (red/blue/green circles/squares/triangles)
- Tests multimodal generation with questions like "What shape is `<image>`?"
- Shows the architecture working (responses are random until trained)
- Optionally runs interactive mode for custom questions

Expected output:
```
🎨 PHASE 4 INTERACTIVE DEMO
Creating test images...
Testing multimodal generation...
Q: What shape is <image>?
A: [random tokens - architecture working but not trained]
✅ PHASE 4 ARCHITECTURE WORKING!
```

### 2. Stage 2 Training Test
```bash
python test_stage2_training.py
```

Validates Stage 2 conversation training:
- Creates shape training data
- Runs mini training loop
- Shows before/after generation
- Validates training pipeline

## 🏃 Quick End-to-End Tests

### 1. Quick Training Test with Real CLIP
```bash
python train_mini_quick.py
```

Ultra-fast test that:
- Uses real CLIP model (downloads ~350MB)
- Trains 1 step of Stage 1 (alignment)
- Trains 1 step of Stage 2 (instruction)
- Tests generation pipeline
- Completes in ~30 seconds

Expected output:
```
🚀 Quick Vision-Language Training Test
✅ Stage 1 works!
✅ Stage 2 works!
✅ Generation works!
🎉 All tests passed!
```

### 2. Full Mini Training (with Real CLIP)
```bash
python train_mini_vision_model.py
```

Complete training pipeline that:
- Downloads and uses real CLIP vision model
- Trains Stage 1: Vision-language alignment (5 epochs)
- Trains Stage 2: Instruction following (5 epochs)
- Creates 50 shape images with descriptions
- Saves checkpoints: `stage1_projector.pt`, `stage2_model.pt`
- Tests on new images

⏱️ Takes ~5-10 minutes on CPU

## 📊 Test Coverage Summary

| Component | Test File | # Tests | Status |
|-----------|-----------|---------|--------|
| Image Tokens | `test_image_tokens.py` | 21 | ✅ |
| Multimodal Inputs | `test_multimodal_inputs.py` | 18 | ✅ |
| Text Generation | `test_text_generation.py` | 14 | ✅ |
| Vision Integration | `tests/test_vision_integration.py` | 15 | ✅ |
| End-to-End | `tests/test_end_to_end_multimodal.py` | 12 | ✅ |
| **Total** | | **80+** | ✅ |

## 🔍 Individual Component Tests

### Test Image Processing
```bash
python test_image_processor_simple.py
```
- Tests image loading and preprocessing
- Validates tensor shapes and formats

### Test Vision Projector
```bash
python tests/test_projector_builder.py
```
- Tests different projector types (linear, MLP)
- Validates dimension transformations

### Test Multimodal Data Pipeline
```bash
python tests/test_multimodal_data.py
```
- Tests data loading and batching
- Validates conversation format processing

## 🐛 Debugging Tests

### Test Imports Only
```bash
python test_imports_only.py
```
Quick sanity check that all imports work correctly.

### Test Trained Model
```bash
python demo_trained_model.py
```
If you have trained checkpoints, this tests loading and inference.

## 📈 Expected Results

### Before Training (Architecture Only)
```python
Q: "What shape is <image>?"
A: "protocols Ethiopia bicycles"  # Random tokens
```

### After Training (With Learned Weights)
```python
Q: "What shape is <image>?"
A: "This is a red triangle."  # Meaningful response
```

## 🎯 Running All Tests

To run all Phase 4 tests in sequence:
```bash
# Core functionality
python test_image_tokens.py
python test_multimodal_inputs.py
python test_text_generation.py

# Integration tests
python demo_phase4_interactive.py
python test_stage2_training.py

# Quick validation with real CLIP
python train_mini_quick.py
```

## ⚠️ Common Issues

### 1. PyTorch not installed
```bash
pip install torch torchvision
```

### 2. CUDA/GPU errors
All tests work on CPU. If you see CUDA errors:
```python
device = 'cpu'  # Force CPU usage
```

### 3. Memory issues
Reduce batch size or model size in config:
```python
config = ModelConfig(
    num_hidden_layers=1,  # Fewer layers
    hidden_size=64,       # Smaller hidden size
    batch_size=2          # Smaller batches
)
```

### 4. CLIP download fails
Ensure you have internet connection and enough disk space (~350MB).

## ✅ Success Criteria

All tests pass when:
1. Architecture processes image tokens correctly
2. Vision features integrate with text embeddings
3. Generation pipeline produces output (even if random)
4. Training loops complete without errors
5. Model can be saved and loaded

## 📚 Documentation

For more details, see:
- `PHASE4_TEXT_GENERATION_PLAN.md` - Implementation plan
- `PHASE4_COMPLETION_SUMMARY.md` - What was built
- `STAGE2_TRAINING_INFRASTRUCTURE.md` - Training details
- `LLAVA_ARCHITECTURE_ANALYSIS.md` - Technical architecture

---

**Note**: These tests validate the complete vision-to-text pipeline is working. Actual meaningful responses require training on real data, but the architecture is proven to work end-to-end!