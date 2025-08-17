#!/usr/bin/env python3
"""
Quick 10K sample training script for RTX 4090
Designed to complete in 1-2 hours for testing
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
from PIL import Image
import time
from datetime import datetime
import os

print("="*60)
print("🚀 Quick 10K Sample Training on RTX 4090")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
else:
    print("⚠️ No GPU found, using CPU (will be slow)")

# Configuration for RTX 4090 (24GB)
config = {
    'num_samples': 10000,  # Quick subset
    'batch_size': 4,       # Conservative for 24GB VRAM
    'learning_rate': 2e-5,
    'num_epochs': 2,       # Quick training
    'save_every': 2500,    # Checkpoint frequency
}

print(f"\n📊 Training Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# ============================================
# 1. LOAD MODEL COMPONENTS
# ============================================

print("\n📦 Loading model components...")

# Your model
from gpt_oss.torch.model import Transformer, ModelConfig
from gpt_oss.constants import DEFAULT_IMAGE_TOKEN

model_config = ModelConfig(
    num_hidden_layers=6,   # Medium size for RTX 4090
    hidden_size=512,
    vocab_size=50258,
    num_attention_heads=8,
    num_key_value_heads=8,
    intermediate_size=2048,
)

model = Transformer(model_config, device='cuda')
print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize image tokenizer
image_token_id = model.initialize_image_tokenizer(tokenizer)
model.set_image_token_id(image_token_id)

# CLIP Vision
print("  Loading CLIP vision encoder...")
try:
    vision_tower = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", 
        use_safetensors=True
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
except:
    print("  Trying fallback method...")
    # Fallback for older PyTorch versions
    import os
    os.environ["TRANSFORMERS_CACHE"] = "/tmp"
    vision_tower = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        trust_remote_code=True
    ).cuda()
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
vision_tower.eval()

# Projector
projector = nn.Sequential(
    nn.Linear(768, model_config.hidden_size),
    nn.GELU(),
    nn.Linear(model_config.hidden_size, model_config.hidden_size)
).cuda()

print("✅ All components loaded!")

# ============================================
# 2. LOAD DATASET (10K subset)
# ============================================

print(f"\n📚 Loading LLaVA dataset (first {config['num_samples']} samples)...")

try:
    # Try to load from HuggingFace
    dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split=f"train[:{config['num_samples']}]")
    print(f"✅ Loaded {len(dataset)} samples from LLaVA-Instruct-150K")
except:
    print("⚠️ Couldn't load LLaVA dataset, using synthetic data")
    # Fallback: Create synthetic data
    dataset = []
    for i in range(config['num_samples']):
        dataset.append({
            'conversations': [
                {'from': 'human', 'value': f'What do you see in {DEFAULT_IMAGE_TOKEN}?'},
                {'from': 'gpt', 'value': f'This is image number {i}.'}
            ],
            'image': None  # Will generate synthetic images
        })

# ============================================
# 3. TRAINING DATASET CLASS
# ============================================

class QuickTrainingDataset(Dataset):
    def __init__(self, data, tokenizer, image_processor):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process conversation
        human_text = ""
        gpt_text = ""
        for conv in item['conversations']:
            if conv['from'] == 'human':
                human_text = conv['value']
            else:
                gpt_text = conv['value']
        
        # Create full text
        full_text = human_text + " " + gpt_text
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_text, max_length=512, truncation=True)
        input_tensor = torch.tensor(input_ids)
        
        # Create labels (mask human part)
        labels = input_tensor.clone()
        human_len = len(self.tokenizer.encode(human_text))
        labels[:human_len] = -100  # Ignore human part in loss
        
        # Process image (synthetic if not available)
        if item.get('image'):
            image = item['image']
        else:
            # Create synthetic image
            image = Image.new('RGB', (224, 224), color=(idx % 255, (idx*2) % 255, (idx*3) % 255))
        
        # Process through CLIP
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        return {
            'input_ids': input_tensor,
            'labels': labels,
            'pixel_values': pixel_values
        }

# Create dataset and dataloader
train_dataset = QuickTrainingDataset(dataset, tokenizer, image_processor)

def collate_fn(batch):
    # Pad sequences
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    pixel_values = []
    
    for item in batch:
        # Pad input_ids
        pad_len = max_len - len(item['input_ids'])
        padded_input = torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([item['labels'], torch.full((pad_len,), -100, dtype=torch.long)])
        
        input_ids.append(padded_input)
        labels.append(padded_labels)
        pixel_values.append(item['pixel_values'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'pixel_values': torch.stack(pixel_values)
    }

train_loader = DataLoader(
    train_dataset, 
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2
)

print(f"✅ Dataset ready: {len(train_loader)} batches")

# ============================================
# 4. TRAINING LOOP
# ============================================

# Setup training
# Wrap vision tower to return tensor directly (fix for transformers output object)
def vision_tower_wrapper(images):
    return vision_tower(images).last_hidden_state

model.vision_tower = vision_tower_wrapper
model.mm_projector = projector

optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(projector.parameters()),
    lr=config['learning_rate']
)

criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Mixed precision for faster training
scaler = torch.cuda.amp.GradScaler()

print(f"\n🎯 Starting training...")
print(f"Estimated time: 1-2 hours on RTX 4090")

start_time = time.time()
global_step = 0

for epoch in range(config['num_epochs']):
    print(f"\n📈 Epoch {epoch+1}/{config['num_epochs']}")
    
    epoch_loss = 0
    batch_times = []
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start = time.time()
        
        # Move to GPU
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        pixel_values = batch['pixel_values'].cuda()
        
        # Mixed precision training
        with torch.amp.autocast('cuda'):
            # Get vision features
            with torch.no_grad():
                vision_features = vision_tower(pixel_values).last_hidden_state
            
            # Project features
            projected_features = projector(vision_features)
            
            # Prepare multimodal inputs
            multimodal_inputs = model.prepare_multimodal_inputs(
                input_ids=input_ids,
                images=pixel_values,
                labels=labels
            )
            
            # Forward pass
            embeddings = multimodal_inputs["inputs_embeds"]
            if embeddings.dtype != torch.bfloat16:
                embeddings = embeddings.to(torch.bfloat16)
            
            hidden_states = model.norm(embeddings)
            logits = model.unembedding(hidden_states).float()
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Tracking
        epoch_loss += loss.item()
        batch_time = time.time() - batch_start
        batch_times.append(batch_time)
        global_step += 1
        
        # Progress reporting
        if (batch_idx + 1) % 100 == 0:
            avg_batch_time = sum(batch_times[-100:]) / len(batch_times[-100:])
            remaining_batches = len(train_loader) - batch_idx
            eta = remaining_batches * avg_batch_time
            
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | "
                  f"Batch/s: {1/avg_batch_time:.2f} | "
                  f"ETA: {eta/60:.1f}min")
        
        # Checkpoint saving
        if global_step % config['save_every'] == 0:
            checkpoint_path = f"checkpoint_step_{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint: {checkpoint_path}")
    
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_epoch_loss:.4f}")

# ============================================
# 5. SAVE FINAL MODEL
# ============================================

total_time = time.time() - start_time
print(f"\n🎉 Training complete in {total_time/60:.1f} minutes!")

# Save final model
final_path = "llava_10k_rtx4090_final.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'projector_state_dict': projector.state_dict(),
    'model_config': model_config,
    'training_config': config,
    'total_time': total_time
}, final_path)

print(f"💾 Final model saved: {final_path}")

# ============================================
# 6. QUICK VALIDATION
# ============================================

print(f"\n🧪 Quick validation...")

model.eval()
projector.eval()

from gpt_oss.generate_multimodal import MultimodalTextGenerator
generator = MultimodalTextGenerator(model, tokenizer)

# Test on a sample
test_image = Image.new('RGB', (224, 224), color=(255, 0, 0))  # Red image
test_prompt = f"What do you see in {DEFAULT_IMAGE_TOKEN}?"

response = generator.generate_response(
    prompt=test_prompt,
    image=test_image,
    max_tokens=20,
    temperature=0.7
)

print(f"Prompt: {test_prompt}")
print(f"Response: {response}")

print("\n✅ Model is generating responses!")
print(f"Total cost estimate: ${(total_time/3600) * 0.34:.2f} on RTX 4090")