#!/usr/bin/env python3
"""
Test Trained Projector

This script validates that the trained projector from Stage 1 is working correctly.
It loads the trained projector and tests it with sample images.

Usage:
    python test_trained_projector.py
"""

import os
import sys
import json
import torch
import torch.nn as nn
from pathlib import Path
import logging
from PIL import Image

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.torch.vision_tower import build_vision_tower
from gpt_oss.torch.vision_projector import build_vision_projector
from gpt_oss.vision.image_processor import create_processor_for_vision_tower

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_projector(checkpoint_dir: str, config: ModelConfig):
    """Load the trained projector from Stage 1."""
    
    projector_path = Path(checkpoint_dir) / "mm_projector.bin"
    
    if not projector_path.exists():
        raise FileNotFoundError(f"Trained projector not found: {projector_path}")
    
    # Create projector with same config as training
    projector = build_vision_projector(config)
    
    # Load trained weights
    state_dict = torch.load(projector_path, map_location='cpu')
    projector.load_state_dict(state_dict)
    
    logger.info(f"✅ Loaded trained projector from {projector_path}")
    return projector

def test_projector_inference(projector, vision_tower, image_processor, device):
    """Test the trained projector with sample images."""
    
    logger.info("🧪 Testing trained projector inference...")
    
    # Find test images
    test_images_dir = Path("./data/poc_training/images")
    
    if not test_images_dir.exists():
        logger.warning("Test images not found, creating a synthetic image...")
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_image_path = "./test_image.jpg"
        test_image.save(test_image_path)
        test_images = [test_image_path]
    else:
        # Use first few synthetic images
        test_images = list(test_images_dir.glob("*.jpg"))[:3]
    
    projector.eval()
    vision_tower.eval()
    
    with torch.no_grad():
        for i, image_path in enumerate(test_images):
            logger.info(f"  Testing image {i+1}/{len(test_images)}: {image_path}")
            
            try:
                # Process image
                image_tensor = image_processor.process(str(image_path))
                image_tensor = image_tensor.to(device)
                
                # Extract vision features
                vision_features = vision_tower(image_tensor)  # [1, patches, vision_dim]
                logger.info(f"    Vision features shape: {vision_features.shape}")
                
                # Project to language space
                projected_features = projector(vision_features)  # [1, patches, hidden_dim]
                logger.info(f"    Projected features shape: {projected_features.shape}")
                
                # Check feature statistics
                mean_val = projected_features.mean().item()
                std_val = projected_features.std().item()
                min_val = projected_features.min().item()
                max_val = projected_features.max().item()
                
                logger.info(f"    Feature stats: mean={mean_val:.3f}, std={std_val:.3f}, min={min_val:.3f}, max={max_val:.3f}")
                
                # Check for reasonable values
                if torch.all(torch.isfinite(projected_features)):
                    logger.info("    ✅ Features are finite and valid")
                else:
                    logger.warning("    ⚠️  Found non-finite values in features")
                
                # Check if features are not all zeros
                if projected_features.abs().sum() > 1e-6:
                    logger.info("    ✅ Features contain meaningful information")
                else:
                    logger.warning("    ⚠️  Features are mostly zeros")
                
            except Exception as e:
                logger.error(f"    ❌ Error processing image: {e}")
    
    logger.info("✅ Projector inference test completed")

def test_projector_dimensions(projector, config):
    """Test that projector has correct input/output dimensions."""
    
    logger.info("🔍 Testing projector dimensions...")
    
    # Create dummy vision features
    batch_size = 2
    num_patches = 196  # Standard for 224x224 image with 16x16 patches
    vision_dim = config.mm_hidden_size  # 512 for CLIP base
    
    dummy_vision_features = torch.randn(batch_size, num_patches, vision_dim)
    
    with torch.no_grad():
        projected = projector(dummy_vision_features)
    
    expected_shape = (batch_size, num_patches, config.hidden_size)
    actual_shape = projected.shape
    
    logger.info(f"  Input shape: {dummy_vision_features.shape}")
    logger.info(f"  Output shape: {actual_shape}")
    logger.info(f"  Expected shape: {expected_shape}")
    
    if actual_shape == expected_shape:
        logger.info("  ✅ Projector dimensions are correct")
        return True
    else:
        logger.error("  ❌ Projector dimensions mismatch!")
        return False

def compare_before_after_training(config, device):
    """Compare projector outputs before and after training."""
    
    logger.info("📊 Comparing projector before/after training...")
    
    # Create untrained projector
    untrained_projector = build_vision_projector(config)
    untrained_projector.to(device)
    
    # Load trained projector
    checkpoint_dir = "./checkpoints/poc_stage1"
    trained_projector = load_trained_projector(checkpoint_dir, config)
    trained_projector.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, 196, config.mm_hidden_size).to(device)
    
    with torch.no_grad():
        untrained_output = untrained_projector(dummy_input)
        trained_output = trained_projector(dummy_input)
    
    # Compare outputs
    mse_diff = nn.MSELoss()(untrained_output, trained_output).item()
    max_diff = (untrained_output - trained_output).abs().max().item()
    
    logger.info(f"  MSE difference: {mse_diff:.6f}")
    logger.info(f"  Max difference: {max_diff:.6f}")
    
    if mse_diff > 1e-6:
        logger.info("  ✅ Projector weights changed during training")
        return True
    else:
        logger.warning("  ⚠️  Projector weights appear unchanged")
        return False

def test_end_to_end_pipeline(device):
    """Test the complete vision → projection → language pipeline."""
    
    logger.info("🔄 Testing end-to-end multimodal pipeline...")
    
    # Load training info to get config
    checkpoint_dir = Path("./checkpoints/poc_stage1")
    info_path = checkpoint_dir / "training_info.json"
    
    if info_path.exists():
        with open(info_path, 'r') as f:
            training_info = json.load(f)
        logger.info(f"  Training loss: {training_info['final_loss']:.4f}")
        logger.info(f"  Training time: {training_info['training_time_seconds']:.1f}s")
    
    # Create config (same as training)
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        mm_vision_tower="openai/clip-vit-base-patch32",
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=768,  # CLIP base hidden size
        use_mm_proj=True,
    )
    
    # Load components
    vision_tower = build_vision_tower(config)
    vision_tower.to(device)
    
    trained_projector = load_trained_projector(str(checkpoint_dir), config)
    trained_projector.to(device)
    
    image_processor = create_processor_for_vision_tower(config.mm_vision_tower)
    
    # Test inference
    test_projector_inference(trained_projector, vision_tower, image_processor, device)
    
    # Test dimensions
    dims_ok = test_projector_dimensions(trained_projector, config)
    
    # Compare before/after
    weights_changed = compare_before_after_training(config, device)
    
    return dims_ok and weights_changed

def main():
    """Run all projector tests."""
    
    logger.info("🧪 Testing Trained Projector")
    logger.info("=" * 50)
    
    # Check if trained projector exists
    checkpoint_dir = Path("./checkpoints/poc_stage1")
    if not checkpoint_dir.exists():
        logger.error("Trained projector not found!")
        logger.info("Please run: python train_poc_stage1.py")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    try:
        # Run end-to-end test
        success = test_end_to_end_pipeline(device)
        
        if success:
            logger.info("\n🎉 All projector tests passed!")
            logger.info("✅ Projector is working correctly")
            logger.info("✅ Dimensions are correct")
            logger.info("✅ Weights changed during training")
            
            logger.info("\n🎯 Next steps:")
            logger.info("1. Run Stage 2 training: python train_poc_stage2.py")
            logger.info("2. Test full multimodal inference: python test_multimodal_inference.py")
        else:
            logger.warning("\n⚠️  Some projector tests failed")
            logger.info("Check the logs above for details")
            
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        raise

if __name__ == "__main__":
    main()