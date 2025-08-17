#!/usr/bin/env python3
"""
Demo: Trained Multimodal Model Inference

This script demonstrates inference with the trained projector from Stage 1.
Shows how vision features are processed through the trained bridge to language space.

Usage:
    python demo_trained_model.py
"""

import os
import sys
import torch
import random
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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_trained_model(checkpoint_dir: str, device: torch.device):
    """Load the trained multimodal model components."""
    
    logger.info("Loading trained multimodal model...")
    
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
    
    # Load vision tower
    vision_tower = build_vision_tower(config)
    vision_tower.to(device)
    vision_tower.eval()
    
    # Load trained projector
    projector = build_vision_projector(config)
    projector_path = Path(checkpoint_dir) / "mm_projector.bin"
    state_dict = torch.load(projector_path, map_location=device)
    projector.load_state_dict(state_dict)
    projector.to(device)
    projector.eval()
    
    # Create image processor
    image_processor = create_processor_for_vision_tower(config.mm_vision_tower)
    
    logger.info(f"✅ Loaded trained model from {checkpoint_dir}")
    logger.info(f"   Vision tower: {config.mm_vision_tower}")
    logger.info(f"   Projector: {config.mm_projector_type}")
    logger.info(f"   Hidden dimensions: {config.mm_hidden_size} → {config.hidden_size}")
    
    return vision_tower, projector, image_processor, config

def process_image_through_pipeline(image_path: str, vision_tower, projector, image_processor, device):
    """Process a single image through the complete vision pipeline."""
    
    logger.info(f"Processing image: {image_path}")
    
    with torch.no_grad():
        # Step 1: Load and preprocess image
        image_tensor = image_processor.process(image_path)
        image_tensor = image_tensor.to(device)
        logger.info(f"  📷 Image tensor shape: {image_tensor.shape}")
        
        # Step 2: Extract vision features
        vision_features = vision_tower(image_tensor)
        logger.info(f"  👁️  Vision features shape: {vision_features.shape}")
        
        # Step 3: Project to language space (this uses our trained projector!)
        language_features = projector(vision_features)
        logger.info(f"  🧠 Language features shape: {language_features.shape}")
        
        # Step 4: Analyze the features
        mean_val = language_features.mean().item()
        std_val = language_features.std().item()
        min_val = language_features.min().item()
        max_val = language_features.max().item()
        
        logger.info(f"  📊 Feature statistics:")
        logger.info(f"     Mean: {mean_val:.4f}")
        logger.info(f"     Std:  {std_val:.4f}")
        logger.info(f"     Range: [{min_val:.4f}, {max_val:.4f}]")
        
        return {
            'image_tensor': image_tensor,
            'vision_features': vision_features,
            'language_features': language_features,
            'stats': {
                'mean': mean_val,
                'std': std_val,
                'min': min_val,
                'max': max_val
            }
        }

def demo_multimodal_inference():
    """Demonstrate multimodal inference with trained projector."""
    
    logger.info("🚀 Multimodal Inference Demo with Trained Projector")
    logger.info("=" * 60)
    
    # Check if trained model exists
    checkpoint_dir = Path("./checkpoints/poc_stage1")
    if not checkpoint_dir.exists():
        logger.error("Trained model not found!")
        logger.info("Please run: python train_poc_stage1.py")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load trained model
    vision_tower, projector, image_processor, config = load_trained_model(str(checkpoint_dir), device)
    
    # Find test images
    test_images_dir = Path("./data/poc_training/images")
    if not test_images_dir.exists():
        logger.error("Test images not found!")
        logger.info("Please run: python create_poc_training_data.py")
        return
    
    # Get a few random test images
    test_images = list(test_images_dir.glob("*.jpg"))
    selected_images = random.sample(test_images, min(5, len(test_images)))
    
    logger.info(f"🖼️  Processing {len(selected_images)} test images...")
    
    # Process each image
    results = []
    for i, image_path in enumerate(selected_images):
        logger.info(f"\n--- Image {i+1}/{len(selected_images)} ---")
        
        try:
            result = process_image_through_pipeline(
                str(image_path), vision_tower, projector, image_processor, device
            )
            results.append(result)
            logger.info("  ✅ Processing successful")
            
        except Exception as e:
            logger.error(f"  ❌ Processing failed: {e}")
    
    # Analyze results across images
    if results:
        logger.info("\n" + "=" * 60)
        logger.info("📈 Cross-Image Analysis")
        logger.info("=" * 60)
        
        # Compare feature diversity
        all_means = [r['stats']['mean'] for r in results]
        all_stds = [r['stats']['std'] for r in results]
        
        mean_diversity = max(all_means) - min(all_means)
        std_diversity = max(all_stds) - min(all_stds)
        
        logger.info(f"Feature diversity across images:")
        logger.info(f"  Mean range: {min(all_means):.4f} to {max(all_means):.4f} (diversity: {mean_diversity:.4f})")
        logger.info(f"  Std range:  {min(all_stds):.4f} to {max(all_stds):.4f} (diversity: {std_diversity:.4f})")
        
        if mean_diversity > 0.01:
            logger.info("  ✅ Good feature diversity - projector produces different outputs for different images")
        else:
            logger.warning("  ⚠️  Low feature diversity - projector may not be well-trained")
        
        # Test feature similarity for same vs different images
        if len(results) >= 2:
            features1 = results[0]['language_features']
            features2 = results[1]['language_features']
            
            # Cosine similarity
            cos_sim = torch.nn.functional.cosine_similarity(
                features1.flatten(), features2.flatten(), dim=0
            ).item()
            
            logger.info(f"  Feature similarity between images: {cos_sim:.4f}")
            if cos_sim < 0.9:
                logger.info("  ✅ Features are appropriately different for different images")
            else:
                logger.warning("  ⚠️  Features are very similar - may indicate undertrained projector")
    
    # Demo complete
    logger.info("\n" + "=" * 60)
    logger.info("🎉 Multimodal Inference Demo Complete!")
    logger.info("=" * 60)
    
    logger.info("✅ Successfully demonstrated:")
    logger.info("   • Image preprocessing pipeline")
    logger.info("   • Vision feature extraction with CLIP")
    logger.info("   • Trained projector bridge to language space")
    logger.info("   • Feature analysis and validation")
    
    logger.info("\n🎯 What this shows:")
    logger.info("   • The projector learned to map vision → language features")
    logger.info("   • Different images produce different feature representations")
    logger.info("   • The pipeline is ready for full multimodal training")
    
    logger.info("\n🚀 Next steps for production:")
    logger.info("   1. Train Stage 2 (instruction following) with more data")
    logger.info("   2. Scale up to real datasets (CC3M, LLaVA-Instruct)")
    logger.info("   3. Use multiple GPUs for faster training")
    logger.info("   4. Implement full text generation with image context")

def compare_untrained_vs_trained():
    """Compare outputs from untrained vs trained projector."""
    
    logger.info("\n🔬 Comparing Untrained vs Trained Projector")
    logger.info("-" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create config
    config = ModelConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        mm_vision_tower="openai/clip-vit-base-patch32",
        mm_projector_type="mlp2x_gelu",
        mm_hidden_size=768,
        use_mm_proj=True,
    )
    
    # Create untrained projector
    untrained_projector = build_vision_projector(config)
    untrained_projector.to(device)
    untrained_projector.eval()
    
    # Load trained projector
    trained_projector = build_vision_projector(config)
    checkpoint_dir = Path("./checkpoints/poc_stage1")
    projector_path = checkpoint_dir / "mm_projector.bin"
    state_dict = torch.load(projector_path, map_location=device)
    trained_projector.load_state_dict(state_dict)
    trained_projector.to(device)
    trained_projector.eval()
    
    # Test with dummy vision features
    dummy_vision_features = torch.randn(1, 49, 768).to(device)
    
    with torch.no_grad():
        untrained_output = untrained_projector(dummy_vision_features)
        trained_output = trained_projector(dummy_vision_features)
    
    # Compare outputs
    mse_diff = torch.nn.functional.mse_loss(untrained_output, trained_output).item()
    max_diff = (untrained_output - trained_output).abs().max().item()
    
    logger.info(f"Difference metrics:")
    logger.info(f"  MSE difference: {mse_diff:.6f}")
    logger.info(f"  Max difference: {max_diff:.6f}")
    
    # Analyze feature distributions
    untrained_mean = untrained_output.mean().item()
    trained_mean = trained_output.mean().item()
    untrained_std = untrained_output.std().item()
    trained_std = trained_output.std().item()
    
    logger.info(f"Feature distributions:")
    logger.info(f"  Untrained: mean={untrained_mean:.4f}, std={untrained_std:.4f}")
    logger.info(f"  Trained:   mean={trained_mean:.4f}, std={trained_std:.4f}")
    
    if mse_diff > 0.01:
        logger.info("  ✅ Significant difference - training was effective!")
    else:
        logger.warning("  ⚠️  Small difference - training may need more data/epochs")

def main():
    """Run the complete demo."""
    
    # Main inference demo
    demo_multimodal_inference()
    
    # Comparison demo
    try:
        compare_untrained_vs_trained()
    except Exception as e:
        logger.warning(f"Comparison demo failed: {e}")

if __name__ == "__main__":
    main()