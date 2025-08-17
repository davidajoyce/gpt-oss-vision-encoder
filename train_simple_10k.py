#!/usr/bin/env python3
"""
Simplified 10K training script that avoids mixed precision and bfloat16 issues
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
from PIL import Image
import time
import numpy as np

print("="*60)
print("🚀 Simple 10K Sample Training (No Mixed Precision)")
print("="*60)

# Check GPU
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    device = 'cuda'
else:
    print("⚠️ No GPU found, using CPU")
    device = 'cpu'

# Simple configuration for stable training
config = {
    'num_samples': 10000,
    'batch_size': 2,        # Smaller batch for stability
    'learning_rate': 1e-5,  # Lower learning rate
    'num_epochs': 2,
    'save_every': 2500,
}

print(f"\n📊 Training Configuration:")
for k, v in config.items():
    print(f"  {k}: {v}")

# Load model components
print("\n📦 Loading model components...")

from gpt_oss.torch.model import Transformer, ModelConfig
from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX

# Smaller model for stability
model_config = ModelConfig(
    num_hidden_layers=4,
    hidden_size=256,        # Smaller
    vocab_size=50258,
    num_attention_heads=8,
    num_key_value_heads=8,
    intermediate_size=1024,
)

model = Transformer(model_config, device=device)
# Convert model to float32 to avoid dtype issues
model = model.float()
print(f"  Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"  Model dtype: {next(model.parameters()).dtype}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# Initialize image tokenizer
image_token_id = model.initialize_image_tokenizer(tokenizer)
model.set_image_token_id(image_token_id)

# CLIP Vision (with fallback)
print("  Loading CLIP vision encoder...")
try:
    vision_tower = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32", 
        use_safetensors=True
    ).to(device)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
except:
    vision_tower = CLIPVisionModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        trust_remote_code=True
    ).to(device)
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

vision_tower.eval()
for param in vision_tower.parameters():
    param.requires_grad = False

# Simple projector (ensure float32)
projector = nn.Linear(768, model_config.hidden_size).to(device).float()

print("✅ All components loaded!")

# Create simple dataset
print(f"\n📚 Creating simple synthetic dataset...")

class SimpleDataset(Dataset):
    def __init__(self, num_samples, tokenizer, image_processor):
        self.num_samples = num_samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        # Create simple synthetic data
        self.data = []
        for i in range(num_samples):
            # Simple conversation
            question = f"What is {DEFAULT_IMAGE_TOKEN}?"
            answer = f"This is image number {i % 100}."
            
            self.data.append({
                'question': question,
                'answer': answer,
                'image_idx': i
            })
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Create synthetic image
        color = (idx % 255, (idx*2) % 255, (idx*3) % 255)
        image = Image.new('RGB', (224, 224), color)
        
        # Process image
        pixel_values = self.image_processor(images=image, return_tensors="pt")['pixel_values'][0]
        
        # Create conversation
        full_text = item['question'] + " " + item['answer']
        input_ids = self.tokenizer.encode(full_text, max_length=512, truncation=True)
        input_tensor = torch.tensor(input_ids)
        
        # Create labels (ignore question part)
        labels = input_tensor.clone()
        question_len = len(self.tokenizer.encode(item['question']))
        labels[:question_len] = IGNORE_INDEX
        
        return {
            'input_ids': input_tensor,
            'labels': labels,
            'pixel_values': pixel_values
        }

# Create dataset
dataset = SimpleDataset(config['num_samples'], tokenizer, image_processor)

def collate_fn(batch):
    max_len = max(len(item['input_ids']) for item in batch)
    
    input_ids = []
    labels = []
    pixel_values = []
    
    for item in batch:
        # Pad sequences
        pad_len = max_len - len(item['input_ids'])
        padded_input = torch.cat([item['input_ids'], torch.zeros(pad_len, dtype=torch.long)])
        padded_labels = torch.cat([item['labels'], torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])
        
        input_ids.append(padded_input)
        labels.append(padded_labels)
        pixel_values.append(item['pixel_values'])
    
    return {
        'input_ids': torch.stack(input_ids),
        'labels': torch.stack(labels),
        'pixel_values': torch.stack(pixel_values)
    }

train_loader = DataLoader(
    dataset, 
    batch_size=config['batch_size'],
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0  # Avoid multiprocessing issues
)

print(f"✅ Dataset ready: {len(train_loader)} batches")

# Setup training
def vision_tower_wrapper(images):
    with torch.no_grad():
        return vision_tower(images).last_hidden_state

model.vision_tower = vision_tower_wrapper
model.mm_projector = projector

# Simple optimizer - no duplicate parameters
optimizer = torch.optim.Adam(
    list(projector.parameters()) + [p for p in model.parameters() if p.requires_grad],
    lr=config['learning_rate']
)

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

print(f"\n🎯 Starting simple training (no mixed precision)...")

model.train()
projector.train()

start_time = time.time()
global_step = 0

for epoch in range(config['num_epochs']):
    print(f"\n📈 Epoch {epoch+1}/{config['num_epochs']}")
    
    epoch_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        
        # Simple forward pass (no autocast)
        try:
            # Prepare multimodal inputs
            multimodal_inputs = model.prepare_multimodal_inputs(
                input_ids=input_ids,
                images=pixel_values,
                labels=labels
            )
            
            # Forward pass - ensure all tensors are float32
            embeddings = multimodal_inputs["inputs_embeds"]
            target_labels = multimodal_inputs["labels"]
            
            # Ensure embeddings are float32
            if embeddings.dtype != torch.float32:
                embeddings = embeddings.float()
            
            # Model forward (model is already float32)
            hidden_states = model.norm(embeddings)
            logits = model.unembedding(hidden_states)
            
            # Ensure logits are float32
            if logits.dtype != torch.float32:
                logits = logits.float()
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_labels.view(-1))
            
            # Simple backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            # Progress
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
            
            # Save checkpoint
            if global_step % config['save_every'] == 0:
                checkpoint_path = f"simple_checkpoint_step_{global_step}.pt"
                torch.save({
                    'step': global_step,
                    'model_state_dict': model.state_dict(),
                    'projector_state_dict': projector.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)
                print(f"  💾 Saved: {checkpoint_path}")
                
        except Exception as e:
            print(f"  ❌ Batch {batch_idx} failed: {e}")
            continue
    
    if num_batches > 0:
        avg_loss = epoch_loss / num_batches
        print(f"✅ Epoch {epoch+1} complete | Avg Loss: {avg_loss:.4f}")

# Save final model
total_time = time.time() - start_time
final_path = "simple_10k_final.pt"
torch.save({
    'model_state_dict': model.state_dict(),
    'projector_state_dict': projector.state_dict(),
    'model_config': model_config,
    'total_time': total_time
}, final_path)

print(f"\n🎉 Training complete!")
print(f"💾 Final model: {final_path}")
print(f"⏱️ Total time: {total_time/60:.1f} minutes")
print(f"💰 Estimated cost: ${(total_time/3600) * 0.34:.2f} on RTX 4090")

# Quick test
print(f"\n🧪 Quick test...")
try:
    from gpt_oss.generate_multimodal import MultimodalTextGenerator
    
    model.eval()
    generator = MultimodalTextGenerator(model, tokenizer)
    
    test_image = Image.new('RGB', (224, 224), 'blue')
    response = generator.generate_response(
        prompt=f"What is {DEFAULT_IMAGE_TOKEN}?",
        image=test_image,
        max_tokens=5
    )
    print(f"Response: {response}")
    print("✅ Model can generate responses!")
except Exception as e:
    print(f"⚠️ Test failed: {e}")

print("✅ Simple training complete!")