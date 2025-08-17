#!/usr/bin/env python3
"""
Mini training script for Stage 1 and Stage 2 with real CLIP model.
Designed to work on small computers with limited resources.
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image, ImageDraw
from transformers import AutoTokenizer, CLIPVisionModel, CLIPImageProcessor
from gpt_oss.torch.model import Transformer, ModelConfig
from gpt_oss.generate_multimodal import MultimodalTextGenerator
from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
import json
from pathlib import Path

# ============================================
# DATASET CREATION
# ============================================

def create_shape_image(shape, color, size=224):
    """Create a simple colored shape image"""
    image = Image.new('RGB', (size, size), 'white')
    draw = ImageDraw.Draw(image)
    
    center = size // 2
    shape_size = size // 3
    
    colors = {'red': (255, 0, 0), 'blue': (0, 0, 255), 'green': (0, 255, 0)}
    color_rgb = colors.get(color, (128, 128, 128))
    
    if shape == 'circle':
        bbox = [center - shape_size, center - shape_size, 
                center + shape_size, center + shape_size]
        draw.ellipse(bbox, fill=color_rgb)
    elif shape == 'square':
        bbox = [center - shape_size, center - shape_size,
                center + shape_size, center + shape_size]
        draw.rectangle(bbox, fill=color_rgb)
    elif shape == 'triangle':
        points = [(center, center - shape_size),
                 (center - shape_size, center + shape_size),
                 (center + shape_size, center + shape_size)]
        draw.polygon(points, fill=color_rgb)
    
    return image

def create_shape_dataset(num_samples=100):
    """Create a small dataset of shape images with descriptions"""
    shapes = ['circle', 'square', 'triangle']
    colors = ['red', 'blue', 'green']
    
    dataset = []
    for _ in range(num_samples):
        shape = np.random.choice(shapes)
        color = np.random.choice(colors)
        
        image = create_shape_image(shape, color)
        
        # Create various descriptions
        descriptions = [
            f"a {color} {shape}",
            f"This is a {color} {shape}.",
            f"The image shows a {color} {shape}.",
            f"{color} {shape}",
            f"A {shape} that is {color}."
        ]
        
        dataset.append({
            'image': image,
            'caption': np.random.choice(descriptions),
            'shape': shape,
            'color': color
        })
    
    return dataset

# ============================================
# VISION COMPONENTS WITH REAL CLIP
# ============================================

class RealVisionTower:
    """Real CLIP vision tower wrapper"""
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        print(f"Loading CLIP vision model: {model_name}")
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name)
        self.hidden_size = self.vision_model.config.hidden_size  # 768 for base
        
        # Freeze vision model - we don't train CLIP
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
        self.vision_model.eval()
    
    def __call__(self, images):
        """Process images through CLIP vision encoder"""
        if isinstance(images, torch.Tensor):
            # Already a tensor, ensure right format
            if images.dim() == 3:
                images = images.unsqueeze(0)
            # Process through CLIP
            with torch.no_grad():
                outputs = self.vision_model(pixel_values=images)
                # Get patch embeddings (not pooled)
                return outputs.last_hidden_state  # [batch, 50, 768] (includes CLS token)
        else:
            # PIL images
            processed = self.image_processor(images=images, return_tensors="pt")
            with torch.no_grad():
                outputs = self.vision_model(**processed)
                return outputs.last_hidden_state

class TrainableProjector(nn.Module):
    """Trainable projector to align CLIP features with language model"""
    def __init__(self, vision_hidden_size, model_hidden_size, projector_type="mlp"):
        super().__init__()
        
        if projector_type == "linear":
            self.projector = nn.Linear(vision_hidden_size, model_hidden_size)
        elif projector_type == "mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_hidden_size, model_hidden_size),
                nn.GELU(),
                nn.Linear(model_hidden_size, model_hidden_size)
            )
        else:
            raise ValueError(f"Unknown projector type: {projector_type}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, vision_features):
        return self.projector(vision_features)

# ============================================
# STAGE 1: VISION-LANGUAGE ALIGNMENT
# ============================================

class Stage1Dataset(Dataset):
    """Dataset for Stage 1 training (image-caption pairs)"""
    def __init__(self, data, tokenizer, image_processor):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image_tensor = self.image_processor(images=item['image'], return_tensors="pt")['pixel_values'][0]
        
        # Tokenize caption
        caption_ids = self.tokenizer.encode(item['caption'], return_tensors="pt")[0]
        
        return {
            'image': image_tensor,
            'caption_ids': caption_ids,
            'caption': item['caption']
        }

def train_stage1(model, tokenizer, vision_tower, projector, train_data, config):
    """
    Stage 1: Train projector to align vision features with text embeddings
    """
    print("\n" + "="*60)
    print("🎯 STAGE 1: VISION-LANGUAGE ALIGNMENT TRAINING")
    print("="*60)
    
    # Create dataset
    dataset = Stage1Dataset(train_data, tokenizer, vision_tower.image_processor)
    
    # Custom collate function to handle variable-length sequences
    def collate_fn(batch):
        images = torch.stack([item['image'] for item in batch])
        
        # Pad caption_ids to same length
        max_len = max(len(item['caption_ids']) for item in batch)
        padded_captions = []
        for item in batch:
            caption = item['caption_ids']
            if len(caption) < max_len:
                padding = torch.zeros(max_len - len(caption), dtype=caption.dtype)
                caption = torch.cat([caption, padding])
            padded_captions.append(caption)
        
        caption_ids = torch.stack(padded_captions)
        
        return {
            'image': images,
            'caption_ids': caption_ids,
            'captions': [item['caption'] for item in batch]
        }
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    
    # Only train the projector
    optimizer = optim.AdamW(projector.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()
    
    projector.train()
    model.eval()  # Language model stays frozen
    
    for epoch in range(config['num_epochs']):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Get vision features from CLIP
            vision_features = vision_tower(batch['image'])  # [batch, 50, 768]
            
            # Project to language model dimension
            projected_features = projector(vision_features)  # [batch, 50, hidden_size]
            
            # Get text embeddings for captions
            with torch.no_grad():
                text_embeds = model.embed_tokens(batch['caption_ids'])  # [batch, seq_len, hidden_size]
            
            # Convert to same dtype (float32 for training)
            projected_features = projected_features.float()
            text_embeds = text_embeds.float()
            
            # Simple alignment loss: match mean pooled features
            vision_pooled = projected_features.mean(dim=1)  # [batch, hidden_size]
            text_pooled = text_embeds.mean(dim=1)  # [batch, hidden_size]
            
            loss = criterion(vision_pooled, text_pooled)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    print("✅ Stage 1 training complete!")
    return projector

# ============================================
# STAGE 2: INSTRUCTION FOLLOWING
# ============================================

class Stage2Dataset(Dataset):
    """Dataset for Stage 2 training (conversations)"""
    def __init__(self, data, tokenizer, image_processor):
        self.data = data
        self.tokenizer = tokenizer
        self.image_processor = image_processor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image_tensor = self.image_processor(images=item['image'], return_tensors="pt")['pixel_values'][0]
        
        # Combine question and answer
        full_text = item['question'] + " " + item['answer']
        input_ids = self.tokenizer.encode(full_text, return_tensors="pt")[0]
        
        # Create labels (mask the question part)
        labels = input_ids.clone()
        question_length = len(self.tokenizer.encode(item['question'], return_tensors="pt")[0])
        labels[:question_length] = IGNORE_INDEX
        
        return {
            'image': image_tensor,
            'input_ids': input_ids,
            'labels': labels
        }

def create_conversation_data(shape_data):
    """Create conversation training data from shape dataset"""
    conversation_data = []
    
    for item in shape_data:
        shape = item['shape']
        color = item['color']
        image = item['image']
        
        # Create various Q&A pairs
        qa_pairs = [
            (f"What shape is {DEFAULT_IMAGE_TOKEN}?", f"This is a {shape}."),
            (f"What color is {DEFAULT_IMAGE_TOKEN}?", f"It is {color}."),
            (f"What do you see in {DEFAULT_IMAGE_TOKEN}?", f"I see a {color} {shape}."),
            (f"Describe {DEFAULT_IMAGE_TOKEN}.", f"This image shows a {color} {shape}."),
            (f"Is this a {shape}?", f"Yes, this is a {shape}."),
            (f"What color is the {shape}?", f"The {shape} is {color}.")
        ]
        
        for question, answer in qa_pairs:
            conversation_data.append({
                'image': image,
                'question': question,
                'answer': answer,
                'shape': shape,
                'color': color
            })
    
    return conversation_data

def train_stage2(model, tokenizer, vision_tower, projector, train_data, config):
    """
    Stage 2: Train model for instruction following with conversations
    """
    print("\n" + "="*60)
    print("🎓 STAGE 2: INSTRUCTION FOLLOWING TRAINING")
    print("="*60)
    
    # Set vision components on model
    model.vision_tower = vision_tower
    model.mm_projector = projector
    
    # Create conversation dataset
    conversation_data = create_conversation_data(train_data)
    dataset = Stage2Dataset(conversation_data[:config['num_samples']], 
                           tokenizer, vision_tower.image_processor)
    
    # Custom collate function for Stage 2
    def collate_fn_stage2(batch):
        images = torch.stack([item['image'] for item in batch])
        
        # Pad input_ids and labels to same length
        max_len = max(len(item['input_ids']) for item in batch)
        padded_inputs = []
        padded_labels = []
        
        for item in batch:
            input_ids = item['input_ids']
            labels = item['labels']
            
            if len(input_ids) < max_len:
                padding_len = max_len - len(input_ids)
                input_padding = torch.zeros(padding_len, dtype=input_ids.dtype)
                label_padding = torch.full((padding_len,), IGNORE_INDEX, dtype=labels.dtype)
                input_ids = torch.cat([input_ids, input_padding])
                labels = torch.cat([labels, label_padding])
            
            padded_inputs.append(input_ids)
            padded_labels.append(labels)
        
        return {
            'image': images,
            'input_ids': torch.stack(padded_inputs),
            'labels': torch.stack(padded_labels)
        }
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn_stage2)
    
    # Train projector + language model (vision tower stays frozen)
    trainable_params = list(projector.parameters()) + list(model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    
    model.train()
    projector.train()
    
    for epoch in range(config['num_epochs']):
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            # Prepare multimodal inputs
            multimodal_inputs = model.prepare_multimodal_inputs(
                input_ids=batch['input_ids'],
                images=batch['image'],
                labels=batch['labels']
            )
            
            # Forward pass
            embeddings = multimodal_inputs["inputs_embeds"]
            labels = multimodal_inputs["labels"]
            
            # Keep embeddings in model's dtype (bfloat16)
            if embeddings.dtype != torch.bfloat16:
                embeddings = embeddings.to(torch.bfloat16)
            
            # Simple forward (you might need to adapt based on your model)
            hidden_states = model.norm(embeddings)
            logits = model.unembedding(hidden_states)
            
            # Convert to float32 only for loss computation
            logits = logits.float()
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{config['num_epochs']}, Loss: {avg_loss:.4f}")
    
    print("✅ Stage 2 training complete!")
    return model

# ============================================
# MAIN TRAINING PIPELINE
# ============================================

def main():
    print("🚀 Mini Vision-Language Model Training")
    print("This will train a small model with real CLIP on simple shape data")
    
    # Configuration for small computers
    stage1_config = {
        'num_epochs': 5,
        'batch_size': 4,
        'learning_rate': 1e-3,
        'num_samples': 50  # Small dataset
    }
    
    stage2_config = {
        'num_epochs': 5,
        'batch_size': 2,
        'learning_rate': 5e-4,
        'num_samples': 100  # Small dataset
    }
    
    # 1. Create small model
    print("\n📦 Setting up model...")
    config = ModelConfig(
        num_hidden_layers=2,  # Very small transformer
        hidden_size=128,       # Small hidden size
        vocab_size=5000,       # Small vocabulary
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=256,
        initial_context_length=512,  # Use this instead of max_position_embeddings
        head_dim=32,  # head_dim = hidden_size / num_attention_heads
        sliding_window=128  # Keep default
    )
    
    model = Transformer(config, device='cpu')
    
    # 2. Load tokenizer
    print("📝 Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except:
        print("Failed to load GPT-2 tokenizer, creating simple tokenizer")
        # Create a simple tokenizer inline
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {}
                self.reverse_vocab = {}
                self.unk_token_id = 1
                self.eos_token_id = 2
                self.pad_token_id = 0
                
            def encode(self, text, return_tensors=None):
                # Simple word-based encoding
                words = text.split()
                ids = []
                for word in words:
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab) + 3
                        self.reverse_vocab[self.vocab[word]] = word
                    ids.append(self.vocab.get(word, self.unk_token_id))
                if return_tensors == "pt":
                    return torch.tensor([ids])
                return ids
            
            def decode(self, ids, skip_special_tokens=False):
                if isinstance(ids, torch.Tensor):
                    ids = ids.tolist()
                return " ".join([self.reverse_vocab.get(i, "<unk>") for i in ids if i > 2 or not skip_special_tokens])
            
            def __len__(self):
                return max(100, len(self.vocab) + 3)
            
            def add_special_tokens(self, tokens_dict):
                return len(tokens_dict.get("additional_special_tokens", []))
            
            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, self.unk_token_id)
        
        tokenizer = SimpleTokenizer()
    
    # Initialize image tokenizer
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    model.set_image_token_id(image_token_id)
    
    # 3. Load real CLIP vision tower
    print("🎨 Loading CLIP vision model...")
    vision_tower = RealVisionTower("openai/clip-vit-base-patch32")
    
    # 4. Create trainable projector
    print("🔗 Creating projector...")
    projector = TrainableProjector(
        vision_hidden_size=vision_tower.hidden_size,  # 768 for CLIP base
        model_hidden_size=model.config.hidden_size,   # 128 for our small model
        projector_type="mlp"
    )
    
    # 5. Create training data
    print("📊 Creating shape dataset...")
    train_data = create_shape_dataset(num_samples=50)
    print(f"Created {len(train_data)} training samples")
    
    # 6. Stage 1 Training
    projector = train_stage1(model, tokenizer, vision_tower, projector, train_data, stage1_config)
    
    # Save Stage 1 checkpoint
    torch.save({
        'projector_state_dict': projector.state_dict(),
        'config': config
    }, 'stage1_projector.pt')
    print("💾 Saved Stage 1 projector to stage1_projector.pt")
    
    # 7. Stage 2 Training  
    model = train_stage2(model, tokenizer, vision_tower, projector, train_data, stage2_config)
    
    # Save Stage 2 checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'projector_state_dict': projector.state_dict(),
        'config': config
    }, 'stage2_model.pt')
    print("💾 Saved Stage 2 model to stage2_model.pt")
    
    # 8. Test the trained model
    print("\n" + "="*60)
    print("🧪 TESTING TRAINED MODEL")
    print("="*60)
    
    model.eval()
    projector.eval()
    
    generator = MultimodalTextGenerator(model, tokenizer)
    
    # Create test images
    test_cases = [
        ('triangle', 'red'),
        ('circle', 'blue'),
        ('square', 'green')
    ]
    
    for shape, color in test_cases:
        print(f"\nTest: {color} {shape}")
        test_image = create_shape_image(shape, color)
        
        questions = [
            f"What shape is {DEFAULT_IMAGE_TOKEN}?",
            f"What color is {DEFAULT_IMAGE_TOKEN}?",
            f"Describe {DEFAULT_IMAGE_TOKEN}."
        ]
        
        for question in questions:
            try:
                response = generator.generate_response(
                    prompt=question,
                    image=test_image,
                    max_tokens=10,
                    temperature=0.1
                )
                print(f"  Q: {question}")
                print(f"  A: {response}")
            except Exception as e:
                print(f"  Error: {e}")
    
    print("\n✅ Training complete! Model can now describe shapes and colors.")
    print("📈 With more training data and epochs, accuracy will improve.")
    
    return model, vision_tower, projector

if __name__ == "__main__":
    try:
        model, vision_tower, projector = main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have installed:")
        print("  pip install torch torchvision transformers pillow")
        import traceback
        traceback.print_exc()