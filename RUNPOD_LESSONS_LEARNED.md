# 🎓 RunPod Training Lessons Learned

Complete guide of debugging lessons learned while training GPT-OSS vision models on RunPod.

## 🚨 **Common Issues & Solutions**

### **Issue 1: Docker Not Available**
**Error**: `docker: command not found`

**Root Cause**: RunPod PyTorch templates don't include Docker by default

**Solution**: Use direct Python approach instead
```bash
# ❌ Don't do this:
docker build -t model . && docker run model

# ✅ Do this instead:
pip install requirements && python train.py
```

**Lesson**: Direct installation is simpler and more reliable on RunPod

---

### **Issue 2: PyTorch Security Vulnerability**
**Error**: 
```
ValueError: Due to a serious vulnerability issue in `torch.load`, even with `weights_only=True`, we now require users to upgrade torch to at least v2.6
```

**Root Cause**: RunPod templates may have older PyTorch versions with security issues

**Solution**: Always upgrade PyTorch first
```bash
pip install --upgrade torch torchvision
```

**Alternative**: Use safetensors when available
```python
model = CLIPVisionModel.from_pretrained(
    "openai/clip-vit-base-patch32", 
    use_safetensors=True  # ✅ Safer loading
)
```

**Lesson**: Upgrade PyTorch before loading any models

---

### **Issue 3: Transformers Output Object vs Tensor**
**Error**: 
```
TypeError: linear(): argument 'input' (position 1) must be Tensor, not BaseModelOutputWithPooling
```

**Root Cause**: Modern transformers return output objects, not raw tensors

**Solution**: Extract tensor from output object
```python
# ❌ Wrong:
features = vision_tower(images)

# ✅ Correct:
def vision_tower_wrapper(images):
    return vision_tower(images).last_hidden_state

model.vision_tower = vision_tower_wrapper
```

**Lesson**: Always wrap vision models to extract the actual tensor

---

### **Issue 4: Mixed Precision + BFloat16 Incompatibility**
**Error**:
```
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'
```

**Root Cause**: Some PyTorch versions have incomplete bfloat16 support with mixed precision

**Solution**: Disable mixed precision for compatibility
```python
# ❌ Problematic:
with torch.cuda.amp.autocast():
    # training code
scaler = torch.cuda.amp.GradScaler()

# ✅ Simple alternative:
# Just use regular float32 training
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

**Lesson**: Skip mixed precision if you encounter compatibility issues

---

### **Issue 5: Batch Size Mismatch**
**Error**:
```
ValueError: Expected input batch_size (256) to match target batch_size (60)
```

**Root Cause**: Multimodal input preparation changes sequence length (adds image tokens)

**Solution**: Use labels from multimodal_inputs, not original labels
```python
# ❌ Wrong:
loss = criterion(logits.view(-1, logits.size(-1)), original_labels.view(-1))

# ✅ Correct:
multimodal_inputs = model.prepare_multimodal_inputs(input_ids, images, labels)
target_labels = multimodal_inputs["labels"]  # Updated sequence length
loss = criterion(logits.view(-1, logits.size(-1)), target_labels.view(-1))
```

**Lesson**: Always use the processed labels from multimodal_inputs

---

### **Issue 6: dtype Mismatch (float32 vs bfloat16)**
**Error**:
```
RuntimeError: expected mat1 and mat2 to have the same dtype, but got: float != c10::BFloat16
```

**Root Cause**: Model weights in bfloat16, input tensors in float32

**Solution**: Force everything to float32
```python
# ✅ Force model to float32
model = model.float()
projector = projector.float()

# ✅ Convert tensors during forward pass
embeddings = embeddings.float()
logits = logits.float()
```

**Lesson**: For training stability, use consistent float32 throughout

---

### **Issue 7: Deprecated API Warnings**
**Warnings**:
```
FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated
FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated
```

**Solution**: Use updated API calls
```python
# ❌ Old API:
with torch.cuda.amp.autocast():
scaler = torch.cuda.amp.GradScaler()

# ✅ New API:
with torch.amp.autocast('cuda'):
scaler = torch.amp.GradScaler('cuda')
```

**Lesson**: Check for API updates when using newer PyTorch versions

---

## 🛠️ **Best Practices Learned**

### **1. Development Workflow**
```bash
# 1. Test locally first
python test_training_script_local.py

# 2. Use simple setup on RunPod
pip install --upgrade torch torchvision datasets transformers pillow

# 3. Start with minimal training
python train_simple_10k.py  # Not the complex version

# 4. Scale up after validation
python train_full_150k.py   # Only after simple works
```

### **2. Error-Resistant Training Script Structure**
```python
def safe_training_loop():
    for batch in dataloader:
        try:
            # Training code here
            loss.backward()
            optimizer.step()
        except Exception as e:
            print(f"Batch failed: {e}")
            continue  # Skip problematic batches
```

### **3. Progressive Debugging**
```python
# Add debug output for tensor shapes
print(f"Debug - Logits: {logits.shape}, Labels: {labels.shape}")

# Validate dtypes
print(f"Model dtype: {next(model.parameters()).dtype}")
print(f"Input dtype: {input_tensor.dtype}")

# Check for NaN/Inf
if torch.isnan(loss):
    print("NaN loss detected!")
```

### **4. Memory-Efficient Configuration**
```python
config = {
    'batch_size': 2,           # Small for compatibility
    'learning_rate': 1e-5,     # Conservative
    'num_workers': 0,          # Avoid multiprocessing issues
    'mixed_precision': False,  # Disable if problems
}
```

## 📊 **RunPod-Specific Optimizations**

### **1. Pod Selection**
- **Testing**: RTX 4090 ($0.34/hr) - 24GB VRAM
- **Production**: A100 40GB ($1.14/hr) - More stable
- **Template**: PyTorch 2.2+ for compatibility

### **2. Storage Setup**
```bash
# Use persistent volumes for checkpoints
Container Disk: 50GB
Volume Disk: 100GB  # For datasets and checkpoints
```

### **3. Dependency Installation**
```bash
# Order matters - install PyTorch first
pip install --upgrade torch torchvision  # Fix security issues
pip install datasets transformers pillow  # Core dependencies  
pip install accelerate                    # Optional optimizations
```

### **4. Cost Optimization**
- Start with 1K samples (15 min, $0.08) to validate
- Scale to 10K samples (2 hr, $0.66) for real testing
- Full 150K training (12 hr, $4-15) only after validation

## 🎯 **Final Working Configuration**

### **Successful RunPod Command**:
```bash
git clone -b djoyce/vision-encoder-llava https://github.com/YOUR_USERNAME/gpt-oss-vision-encoder.git && cd gpt-oss-vision-encoder && pip install --upgrade torch torchvision datasets transformers pillow accelerate && python train_simple_10k.py
```

### **Key Success Factors**:
1. ✅ **Upgraded PyTorch** (security + compatibility)
2. ✅ **float32 everywhere** (no dtype mismatches)
3. ✅ **No mixed precision** (avoid compatibility issues)
4. ✅ **Vision wrapper** (extract tensors from objects)
5. ✅ **Proper label handling** (use multimodal_inputs labels)
6. ✅ **Error handling** (skip failed batches)
7. ✅ **Conservative settings** (small batch size, low LR)

### **Expected Results**:
- Loss starts ~10.5, decreases to ~3-6
- Training completes in ~2 hours
- Cost: ~$0.66 on RTX 4090
- Output: Trained vision-language model

## 🚀 **Scaling Up Recommendations**

After successful 10K training:

1. **Increase dataset size**: 10K → 50K → 150K
2. **Upgrade GPU**: RTX 4090 → A100 40GB → A100 80GB
3. **Optimize batch size**: 2 → 4 → 8 (as memory allows)
4. **Add mixed precision**: Only after everything works in float32
5. **Use real datasets**: Replace synthetic with LLaVA-Instruct-150K

## 📚 **Key Takeaways**

1. **Start Simple**: Get basic training working before optimizing
2. **Debug Systematically**: Fix one issue at a time
3. **Test Locally**: Validate components before deploying
4. **Use Float32**: Avoid dtype issues during development
5. **Handle Errors Gracefully**: Skip problematic batches
6. **Monitor Progress**: Add debug output and checkpoints
7. **Scale Gradually**: 1K → 10K → 150K samples

## 💡 **When Things Go Wrong**

1. **Check PyTorch version**: `python -c "import torch; print(torch.__version__)"`
2. **Verify GPU memory**: `nvidia-smi`
3. **Test components individually**: Run validation scripts
4. **Simplify configuration**: Reduce batch size, disable optimizations
5. **Add debug output**: Print tensor shapes and dtypes
6. **Start over**: Sometimes a fresh pod with clean environment helps

---

**Remember**: Every error is a learning opportunity. The debugging process taught us how to build robust, production-ready training pipelines that work reliably across different environments! 🎓