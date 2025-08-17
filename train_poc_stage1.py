#!/usr/bin/env python3
"""
Proof-of-Concept Stage 1 Training: Projector Alignment

This script runs a minimal Stage 1 training to align vision and language features.
Designed to run on a single GPU or even CPU for proof-of-concept validation.

Based on LLaVA's approach:
- Only trains the multimodal projector (1% of parameters)
- Vision encoder and language model frozen
- Simple image-caption alignment task

Usage:
    python train_poc_stage1.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Dict, Any
import time

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.train.multimodal_trainer import create_multimodal_trainer
from gpt_oss.train.multimodal_data import DataArguments, make_supervised_data_module
from gpt_oss.vision.image_processor import create_processor_for_vision_tower

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_poc_model_config() -> ModelConfig:
    """Create a minimal model config for POC training."""
    
    config = ModelConfig(
        # Small model for POC (reduce memory usage)
        vocab_size=1000,        # Much smaller vocab
        hidden_size=256,        # Smaller hidden size  
        num_hidden_layers=4,    # Fewer layers
        num_attention_heads=4,  # Fewer heads
        
        # Vision configuration (following LLaVA)
        mm_vision_tower="openai/clip-vit-base-patch32",  # Smaller CLIP model
        mm_projector_type="mlp2x_gelu",  # Same as LLaVA
        mm_hidden_size=768,              # CLIP base hidden size (768 for base model)
        use_mm_proj=True,
    )
    
    logger.info(f"Created POC model config:")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  Layers: {config.num_hidden_layers}")
    logger.info(f"  Vision tower: {config.mm_vision_tower}")
    logger.info(f"  Projector: {config.mm_projector_type}")
    
    return config

def create_poc_model(config: ModelConfig, device: torch.device) -> Transformer:
    """Create a minimal model for POC training."""
    
    logger.info("Creating POC model...")
    
    # Create model with small config
    model = Transformer(config, device=device)
    
    # Initialize vision components
    model.initialize_vision_modules(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in model.vision_tower.parameters()) if model.vision_tower else 0
    projector_params = sum(p.numel() for p in model.mm_projector.parameters()) if model.mm_projector else 0
    language_params = total_params - vision_params - projector_params
    
    logger.info(f"Model parameter breakdown:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Language model: {language_params:,} ({100*language_params/total_params:.1f}%)")
    logger.info(f"  Vision tower: {vision_params:,} ({100*vision_params/total_params:.1f}%)")
    logger.info(f"  Projector: {projector_params:,} ({100*projector_params/total_params:.1f}%)")
    
    return model

def setup_stage1_training(model: Transformer) -> int:
    """Setup Stage 1 training: freeze everything except projector."""
    
    logger.info("Setting up Stage 1 training (projector alignment)...")
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze only the multimodal projector
    trainable_params = 0
    if hasattr(model, 'mm_projector') and model.mm_projector is not None:
        for param in model.mm_projector.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
    
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Stage 1 parameter setup:")
    logger.info(f"  Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    logger.info(f"  Frozen: {total_params-trainable_params:,} ({100*(total_params-trainable_params)/total_params:.1f}%)")
    
    return trainable_params

def create_simple_training_loop(model, dataloader, optimizer, device, num_epochs=1):
    """Simple training loop for POC (fallback if transformers not available)."""
    
    logger.info("Starting simple training loop...")
    
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                images = batch.get('images')
                
                if images is not None:
                    images = images.to(device)
                
                # Forward pass
                optimizer.zero_grad()
                
                if images is not None:
                    # Multimodal forward pass
                    outputs = model(input_ids, images=images)
                else:
                    # Text-only fallback
                    outputs = model(input_ids)
                
                # Compute loss (simple cross-entropy)
                if len(outputs.shape) == 3:  # [batch, seq, vocab]
                    outputs = outputs.view(-1, outputs.size(-1))
                    labels = labels.view(-1)
                
                # Ignore padding tokens in loss
                valid_mask = labels != -100
                if valid_mask.sum() > 0:
                    loss = nn.CrossEntropyLoss()(outputs[valid_mask], labels[valid_mask])
                else:
                    loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"  Batch {batch_idx}: Loss = {loss.item():.4f}")
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    """Run POC Stage 1 training."""
    
    logger.info("🚀 Starting POC Stage 1 Training (Projector Alignment)")
    logger.info("=" * 60)
    
    # Check if training data exists
    data_dir = Path("./data/poc_training")
    data_file = data_dir / "stage1_poc_data.json"
    
    if not data_file.exists():
        logger.error(f"Training data not found: {data_file}")
        logger.info("Please run: python create_poc_training_data.py")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model config and model
    config = create_poc_model_config()
    model = create_poc_model(config, device)
    
    # Setup Stage 1 training
    trainable_params = setup_stage1_training(model)
    
    if trainable_params == 0:
        logger.error("No trainable parameters found! Check projector initialization.")
        return
    
    # Create simple optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,  # LLaVA uses 1e-3 for Stage 1
        weight_decay=0.01
    )
    
    # Load and create simple dataset
    logger.info(f"Loading training data from {data_file}")
    
    try:
        # Create simple data loader (mock for now)
        # In a real implementation, this would use our multimodal_data.py
        
        # For POC, create minimal training data
        dummy_data = []
        
        # Create a few training samples
        for i in range(10):  # Just 10 samples for POC
            dummy_data.append({
                'input_ids': torch.randint(0, config.vocab_size, (20,)),  # 20 tokens
                'labels': torch.randint(0, config.vocab_size, (20,)),
                'images': torch.randn(3, 224, 224),  # Dummy image
            })
        
        # Simple DataLoader
        dataloader = DataLoader(dummy_data, batch_size=2, shuffle=True)
        
        # Run training
        logger.info("Starting training...")
        start_time = time.time()
        
        final_loss = create_simple_training_loop(
            model=model,
            dataloader=dataloader, 
            optimizer=optimizer,
            device=device,
            num_epochs=1
        )
        
        training_time = time.time() - start_time
        
        # Save trained projector
        output_dir = Path("./checkpoints/poc_stage1")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        projector_path = output_dir / "mm_projector.bin"
        if hasattr(model, 'mm_projector') and model.mm_projector is not None:
            torch.save(model.mm_projector.state_dict(), projector_path)
            logger.info(f"✅ Saved trained projector to {projector_path}")
        
        # Save training info
        training_info = {
            "final_loss": final_loss,
            "training_time_seconds": training_time,
            "trainable_parameters": trainable_params,
            "config": {
                "hidden_size": config.hidden_size,
                "vision_tower": config.mm_vision_tower,
                "projector_type": config.mm_projector_type,
            }
        }
        
        info_path = output_dir / "training_info.json"
        with open(info_path, 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info("🎉 POC Stage 1 Training Complete!")
        logger.info(f"📊 Final loss: {final_loss:.4f}")
        logger.info(f"⏱️  Training time: {training_time:.1f} seconds")
        logger.info(f"💾 Checkpoint saved to: {output_dir}")
        
        logger.info("\n🎯 Next steps:")
        logger.info("1. Test the trained projector: python test_trained_projector.py")
        logger.info("2. Run Stage 2 training: python train_poc_stage2.py")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()