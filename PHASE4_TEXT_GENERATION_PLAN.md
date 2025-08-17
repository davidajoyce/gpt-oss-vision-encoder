# Phase 4: Vision-to-Text Generation Implementation Plan

## Executive Summary

This document outlines the implementation plan for **Phase 4: Vision-to-Text Generation**, the final phase that will enable our multimodal vision system to generate natural language descriptions of visual content. Based on comprehensive analysis of LLaVA's architecture, this phase will implement image token handling and autoregressive text generation to complete the vision-to-language pipeline.

**Goal**: Transform our current feature-level shape recognition into natural language responses like "This is a red triangle."

---

## 🎯 **PHASE 4 OBJECTIVES**

### **Primary Goal**
Enable the model to generate text descriptions of visual content by implementing:
1. **Image token integration** in text sequences
2. **Autoregressive text generation** with multimodal context
3. **Instruction following** with visual inputs
4. **Stage 2 training** with conversation data

### **Target Capabilities**
```python
# Current state (Phase 3)
image = load_image("triangle.jpg")
features = model.extract_features(image)  # [49, 256] feature tensor
print(features.shape)  # torch.Size([49, 256])

# Target state (Phase 4) 
image = load_image("triangle.jpg")
response = model.chat("What shape is this?", image=image)
print(response)  # "This is a red triangle."
```

---

## 📋 **IMPLEMENTATION ROADMAP**

### **Task 1: Image Token Infrastructure** 
**Priority**: Critical  
**Estimated Time**: 2-3 hours  
**Dependencies**: None  

#### **1.1 Add Image Token Constants**
```python
# Add to gpt_oss/constants.py (new file)
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
IGNORE_INDEX = -100

# Update tokenizer configuration
def add_image_tokens(tokenizer):
    tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN]
    })
    return tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
```

#### **1.2 Extend Tokenizer Integration**
```python
# Update gpt_oss/torch/model.py
class ExtendedTransformer:
    def initialize_image_tokenizer(self, tokenizer):
        # Add <image> token to vocabulary
        # Resize embeddings if needed
        # Set IMAGE_TOKEN_INDEX
```

#### **Tests**:
- Token addition and vocabulary extension
- Embedding resizing validation
- Token ID consistency checks

---

### **Task 2: Multimodal Input Preparation**
**Priority**: Critical  
**Estimated Time**: 4-5 hours  
**Dependencies**: Task 1  

#### **2.1 Implement prepare_multimodal_inputs()**
Based on LLaVA's `prepare_inputs_labels_for_multimodal()`:

```python
# Add to gpt_oss/torch/model.py
def prepare_multimodal_inputs(self, input_ids, images=None, labels=None):
    """
    Replace <image> tokens in input_ids with actual image features
    
    Args:
        input_ids: [batch, seq_len] text tokens including <image> placeholders
        images: [batch, 3, 224, 224] or None
        labels: [batch, seq_len] or None for training
        
    Returns:
        inputs_embeds: [batch, new_seq_len, hidden_dim] combined embeddings
        new_labels: [batch, new_seq_len] updated labels for training
        attention_mask: [batch, new_seq_len] attention mask
    """
    if images is None:
        # Text-only path (backward compatibility)
        return self.embed_tokens(input_ids), labels, None
    
    # 1. Encode images to features
    image_features = self.encode_images(images)  # [batch, num_patches, hidden_dim]
    
    # 2. Find <image> token positions
    image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)
    
    # 3. Split text around image tokens
    new_input_embeds = []
    new_labels = []
    
    for batch_idx in range(input_ids.shape[0]):
        cur_input_ids = input_ids[batch_idx]
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        
        if num_images == 0:
            # No images in this sequence
            text_embeds = self.embed_tokens(cur_input_ids)
            new_input_embeds.append(text_embeds)
            if labels is not None:
                new_labels.append(labels[batch_idx])
            continue
            
        # Split sequence at image token positions
        image_positions = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
        text_segments = []
        label_segments = []
        
        # Get text segments between image tokens
        prev_pos = 0
        for img_pos in image_positions + [len(cur_input_ids)]:
            if img_pos > prev_pos:
                text_segments.append(cur_input_ids[prev_pos:img_pos])
                if labels is not None:
                    label_segments.append(labels[batch_idx][prev_pos:img_pos])
            prev_pos = img_pos + 1
        
        # Build combined embedding sequence
        combined_embeds = []
        combined_labels = []
        
        for i, text_seg in enumerate(text_segments):
            # Add text segment
            if len(text_seg) > 0:
                text_embeds = self.embed_tokens(text_seg)
                combined_embeds.append(text_embeds)
                if labels is not None:
                    combined_labels.append(label_segments[i])
            
            # Add image features (except after last text segment)
            if i < len(text_segments) - 1:
                img_features = image_features[batch_idx]  # [num_patches, hidden_dim]
                combined_embeds.append(img_features)
                if labels is not None:
                    # Image tokens should be ignored in loss
                    img_labels = torch.full(
                        (img_features.shape[0],), 
                        IGNORE_INDEX, 
                        device=labels.device, 
                        dtype=labels.dtype
                    )
                    combined_labels.append(img_labels)
        
        # Concatenate all segments
        final_embeds = torch.cat(combined_embeds, dim=0)
        new_input_embeds.append(final_embeds)
        
        if labels is not None:
            final_labels = torch.cat(combined_labels, dim=0)
            new_labels.append(final_labels)
    
    # Pad sequences to same length
    return self.pad_sequences(new_input_embeds, new_labels)
```

#### **2.2 Add Sequence Padding Utility**
```python
def pad_sequences(self, input_embeds_list, labels_list=None):
    """Pad variable-length sequences to same length"""
    max_len = max(x.shape[0] for x in input_embeds_list)
    batch_size = len(input_embeds_list)
    hidden_dim = input_embeds_list[0].shape[1]
    
    # Create padded tensors
    padded_embeds = torch.zeros(batch_size, max_len, hidden_dim)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    
    if labels_list is not None:
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX)
    
    for i, embeds in enumerate(input_embeds_list):
        seq_len = embeds.shape[0]
        padded_embeds[i, :seq_len] = embeds
        attention_mask[i, :seq_len] = True
        
        if labels_list is not None:
            padded_labels[i, :seq_len] = labels_list[i]
    
    result = {"inputs_embeds": padded_embeds, "attention_mask": attention_mask}
    if labels_list is not None:
        result["labels"] = padded_labels
    
    return result
```

#### **Tests**:
- Single image, single text segment
- Multiple images in one sequence  
- Text-only sequences (backward compatibility)
- Batch processing with mixed content
- Label alignment for training

---

### **Task 3: Autoregressive Text Generation**
**Priority**: Critical  
**Estimated Time**: 5-6 hours  
**Dependencies**: Task 2  

#### **3.1 Extend Generation Pipeline**
```python
# Update gpt_oss/generate_multimodal.py
class MultimodalTextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
    
    def generate_response(self, prompt, image=None, max_tokens=50, temperature=0.7):
        """
        Generate text response with optional image context
        
        Args:
            prompt: str, e.g., "What shape is this? <image>"
            image: PIL.Image, torch.Tensor, or None
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            str: Generated response
        """
        # 1. Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        
        # 2. Prepare multimodal inputs
        if image is not None:
            # Process image
            if isinstance(image, str):
                image = Image.open(image)
            image_tensor = self.process_image(image)
            
            # Prepare combined input
            multimodal_inputs = self.model.prepare_multimodal_inputs(
                input_ids=input_ids,
                images=image_tensor.unsqueeze(0)
            )
        else:
            # Text-only generation
            multimodal_inputs = {
                "inputs_embeds": self.model.embed_tokens(input_ids),
                "attention_mask": torch.ones_like(input_ids, dtype=torch.bool)
            }
        
        # 3. Generate tokens autoregressively
        generated_ids = self.autoregressive_generate(
            multimodal_inputs, max_tokens, temperature
        )
        
        # 4. Decode to text
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response
    
    def autoregressive_generate(self, initial_inputs, max_tokens, temperature):
        """Generate tokens one by one"""
        current_embeds = initial_inputs["inputs_embeds"]
        attention_mask = initial_inputs["attention_mask"]
        generated_ids = []
        
        for _ in range(max_tokens):
            # Forward pass
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask
                )
                logits = outputs.logits[:, -1, :]  # Last token logits
            
            # Sample next token
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            generated_ids.append(next_token.item())
            
            # Stop if EOS token
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Update inputs for next iteration
            next_embed = self.model.embed_tokens(next_token)
            current_embeds = torch.cat([current_embeds, next_embed], dim=1)
            attention_mask = torch.cat([
                attention_mask, 
                torch.ones(1, 1, dtype=torch.bool)
            ], dim=1)
        
        return generated_ids
```

#### **3.2 Add Image Processing Utilities**
```python
def process_image(self, image):
    """Convert PIL Image to model input tensor"""
    if isinstance(image, str):
        image = Image.open(image)
    
    # Use existing image processor
    from gpt_oss.vision.image_processor import ImageProcessor
    processor = ImageProcessor()
    return processor.process_single_image(image)
```

#### **Tests**:
- Text-only generation (backward compatibility)
- Single image + text generation
- Generation with different sampling parameters
- EOS token handling
- Maximum length enforcement

---

### **Task 4: Stage 2 Training Infrastructure** 
**Priority**: High  
**Estimated Time**: 3-4 hours  
**Dependencies**: Tasks 1-3  

#### **4.1 Conversation Data Format**
```python
# Add to gpt_oss/train/multimodal_data.py
class InstructionDataset(Dataset):
    """Dataset for Stage 2 instruction following training"""
    
    def __init__(self, data_path, tokenizer, image_processor):
        self.conversations = self.load_conversation_data(data_path)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def load_conversation_data(self, path):
        """
        Load conversation data in format:
        {
            "conversations": [
                {"from": "human", "value": "<image>\nWhat shape is this?"},
                {"from": "gpt", "value": "This is a red triangle."}
            ],
            "image": "path/to/image.jpg"
        }
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def __getitem__(self, idx):
        conv = self.conversations[idx]
        
        # Build conversation text
        text = ""
        for turn in conv["conversations"]:
            if turn["from"] == "human":
                text += f"USER: {turn['value']} "
            else:
                text += f"ASSISTANT: {turn['value']}"
        
        # Tokenize
        tokens = self.tokenizer.encode(text, return_tensors="pt")
        
        # Load image if present
        image = None
        if "image" in conv:
            image = self.image_processor.load_image(conv["image"])
        
        return {
            "input_ids": tokens.squeeze(),
            "image": image,
            "labels": tokens.squeeze()  # For training, labels = input_ids
        }
```

#### **4.2 Update Multimodal Trainer for Stage 2**
```python
# Update gpt_oss/train/multimodal_trainer.py
def train_stage2(self, instruction_data_path, num_epochs=3):
    """
    Stage 2 training: Full model fine-tuning for instruction following
    """
    print("=== Starting Stage 2: Instruction Following Training ===")
    
    # 1. Unfreeze all parameters
    for param in self.model.parameters():
        param.requires_grad = True
    
    # 2. Lower learning rate for pre-trained components
    optimizer = self.create_stage2_optimizer()
    
    # 3. Create instruction dataset
    dataset = InstructionDataset(
        instruction_data_path, 
        self.tokenizer, 
        self.image_processor
    )
    dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    # 4. Training loop with multimodal inputs
    for epoch in range(num_epochs):
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            self.optimizer.zero_grad()
            
            # Prepare multimodal inputs
            multimodal_inputs = self.model.prepare_multimodal_inputs(
                input_ids=batch["input_ids"],
                images=batch["image"],
                labels=batch["labels"]
            )
            
            # Forward pass
            outputs = self.model(**multimodal_inputs)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} complete. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        self.save_checkpoint(f"stage2_epoch_{epoch}")

def create_stage2_optimizer(self):
    """Create optimizer with different learning rates for different components"""
    # Lower LR for pre-trained vision tower
    vision_params = list(self.model.vision_tower.parameters())
    
    # Higher LR for projector and language model
    other_params = [p for p in self.model.parameters() 
                   if p not in vision_params]
    
    return torch.optim.AdamW([
        {"params": vision_params, "lr": 2e-6},      # Lower for pre-trained
        {"params": other_params, "lr": 2e-5}       # Higher for new components
    ])
```

#### **Tests**:
- Conversation data loading and processing
- Multimodal batch creation
- Stage 2 training loop
- Checkpoint saving/loading
- Learning rate scheduling

---

### **Task 5: Integration and Validation**
**Priority**: Medium  
**Estimated Time**: 2-3 hours  
**Dependencies**: Tasks 1-4  

#### **5.1 End-to-End Integration Test**
```python
# Create test_phase4_integration.py
def test_full_pipeline():
    """Test complete vision-to-text pipeline"""
    
    # 1. Load trained model
    model = load_multimodal_model("checkpoints/stage2_final.pth")
    tokenizer = load_tokenizer_with_image_tokens()
    generator = MultimodalTextGenerator(model, tokenizer)
    
    # 2. Test with different shapes
    test_cases = [
        ("triangle.jpg", "What shape is this?"),
        ("circle.jpg", "Describe this image."),
        ("square.jpg", "What do you see?")
    ]
    
    for image_path, prompt in test_cases:
        full_prompt = f"{prompt} <image>"
        response = generator.generate_response(full_prompt, image_path)
        print(f"Image: {image_path}")
        print(f"Prompt: {prompt}")
        print(f"Response: {response}")
        print("---")

def test_backward_compatibility():
    """Ensure text-only generation still works"""
    generator = MultimodalTextGenerator(model, tokenizer)
    response = generator.generate_response("What is 2+2?", image=None)
    assert "4" in response
```

#### **5.2 Performance Benchmarking**
```python
# Create benchmark_phase4.py
def benchmark_generation_speed():
    """Measure generation speed with and without images"""
    
    import time
    
    # Text-only baseline
    start = time.time()
    for _ in range(100):
        response = generator.generate_response("Hello", image=None)
    text_only_time = time.time() - start
    
    # Multimodal generation
    start = time.time() 
    for _ in range(100):
        response = generator.generate_response("What is this? <image>", "test.jpg")
    multimodal_time = time.time() - start
    
    print(f"Text-only: {text_only_time:.2f}s (100 generations)")
    print(f"Multimodal: {multimodal_time:.2f}s (100 generations)")
    print(f"Overhead: {(multimodal_time/text_only_time - 1)*100:.1f}%")
```

---

## 🎯 **TESTING STRATEGY**

### **Unit Tests** (Task-by-Task)
- **Task 1**: Token integration and vocabulary tests
- **Task 2**: Multimodal input preparation tests
- **Task 3**: Generation pipeline tests  
- **Task 4**: Training infrastructure tests
- **Task 5**: Integration and performance tests

### **Integration Tests** (End-to-End)
```python
test_files = [
    "test_image_tokens.py",           # Task 1 validation
    "test_multimodal_inputs.py",      # Task 2 validation  
    "test_text_generation.py",       # Task 3 validation
    "test_stage2_training.py",       # Task 4 validation
    "test_phase4_integration.py"     # Task 5 validation
]
```

### **Validation Criteria**
- ✅ Generate coherent text responses to visual inputs
- ✅ Maintain backward compatibility with text-only generation
- ✅ Handle missing images gracefully
- ✅ Process various image formats and sizes
- ✅ Demonstrate shape recognition in generated text
- ✅ Complete Stage 2 training without errors

---

## 📊 **SUCCESS METRICS**

### **Functional Metrics**
- **Shape Recognition**: Model correctly identifies shapes in generated text
- **Instruction Following**: Responds appropriately to "What is this?" prompts
- **Coherence**: Generated text is grammatically correct and relevant
- **Consistency**: Similar images produce similar descriptions

### **Performance Metrics**  
- **Generation Speed**: <2x overhead compared to text-only generation
- **Memory Usage**: No significant memory leaks during generation
- **Training Stability**: Stage 2 training converges without errors
- **Error Handling**: Graceful handling of corrupted/missing images

### **Target Outputs**
```python
# Expected successful outputs
test_triangle = generator.generate_response("What shape is this? <image>", "triangle.jpg")
assert "triangle" in test_triangle.lower()

test_circle = generator.generate_response("Describe this image. <image>", "circle.jpg") 
assert "circle" in test_circle.lower()

test_color = generator.generate_response("What color is this shape? <image>", "red_square.jpg")
assert "red" in test_color.lower() and "square" in test_color.lower()
```

---

## 🚀 **IMPLEMENTATION TIMELINE**

### **Phase 4A: Core Infrastructure** (Week 1)
- **Day 1-2**: Task 1 (Image tokens)
- **Day 3-4**: Task 2 (Multimodal inputs)  
- **Day 5-7**: Task 3 (Text generation)

### **Phase 4B: Training & Integration** (Week 2)
- **Day 1-3**: Task 4 (Stage 2 training)
- **Day 4-5**: Task 5 (Integration testing)
- **Day 6-7**: Performance optimization and documentation

### **Milestones**
- **Milestone 1**: Image tokens successfully integrated
- **Milestone 2**: Multimodal input preparation working
- **Milestone 3**: Basic text generation with images
- **Milestone 4**: Stage 2 training completed
- **Milestone 5**: Full pipeline generating shape descriptions

---

## 🎯 **EXPECTED OUTCOMES**

### **Before Phase 4** (Current State)
```python
image = load_image("triangle.jpg")
features = model.extract_features(image)
print(features.shape)  # torch.Size([49, 256])
# Model sees triangle patterns but can't describe them
```

### **After Phase 4** (Target State)
```python
generator = MultimodalTextGenerator(model, tokenizer)

# Basic shape recognition
response = generator.generate_response("What shape is this? <image>", "triangle.jpg")
print(response)  # "This is a red triangle."

# Color and shape together  
response = generator.generate_response("Describe this image. <image>", "blue_circle.jpg")
print(response)  # "This is a blue circle."

# Open-ended questions
response = generator.generate_response("What do you see? <image>", "green_square.jpg")
print(response)  # "I see a green square shape."
```

---

## 🔄 **ITERATIVE IMPROVEMENT PLAN**

### **Phase 4.1: Basic Shape Description** (This implementation)
- Simple shape and color recognition
- Single image, single question format
- Basic conversation structure

### **Phase 4.2: Enhanced Capabilities** (Future)
- Multiple objects in one image
- Spatial relationships ("The triangle is above the circle")
- More complex visual reasoning

### **Phase 4.3: Advanced Features** (Future)
- Multiple images in one conversation
- Complex instruction following
- Visual question answering

---

## 📁 **DELIVERABLES**

### **New Files to Create**
```
gpt_oss/constants.py                    # Image token constants
gpt_oss/train/instruction_dataset.py    # Stage 2 data handling
test_phase4_integration.py             # Integration tests
benchmark_phase4.py                    # Performance testing
demo_vision_to_text.py                 # Working demonstration
```

### **Files to Update**  
```
gpt_oss/torch/model.py                 # Add multimodal input preparation
gpt_oss/generate_multimodal.py         # Add text generation
gpt_oss/train/multimodal_trainer.py    # Add Stage 2 training
VISION_IMPLEMENTATION_PLAN.md          # Add Phase 4 completion
```

### **Documentation Updates**
```
MULTIMODAL_STATE_DOCUMENT.md           # Update with Phase 4 capabilities
README.md                             # Add vision-to-text examples
MULTIMODAL_AI_LEARNING_GUIDE.md       # Add Stage 2 training guide
```

---

## 🎯 **SUCCESS DEFINITION**

**Phase 4 is successful when:**

✅ **The model generates text descriptions of visual content**  
✅ **Shape recognition works**: "This is a triangle"  
✅ **Color recognition works**: "This is red"  
✅ **Combined recognition works**: "This is a red triangle"  
✅ **Instruction following works**: Responds to "What is this?" appropriately  
✅ **Backward compatibility maintained**: Text-only generation unaffected  
✅ **Training pipeline complete**: Stage 2 training succeeds  
✅ **Integration stable**: No performance regressions  

**Bottom Line**: Our model will finally be able to **speak about what it sees** instead of just seeing it silently at the feature level.

---

*This plan builds directly on the solid foundation of Phases 1-3 and will complete the vision-to-language translation capability that matches and extends LLaVA's approach within the GPT-OSS architecture.*