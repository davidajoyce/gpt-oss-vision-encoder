#!/usr/bin/env python3
"""
Interactive Multimodal Demo

Feed the trained model any image and see how it processes it through the
complete vision→language pipeline. This demonstrates the "understanding"
at the feature level.

Usage:
    python interactive_demo.py [image_path]
    
    # Or run interactively:
    python interactive_demo.py
    # Then enter image paths when prompted
"""

import os
import sys
import torch
import argparse
from pathlib import Path
import logging
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.torch.vision_tower import build_vision_tower
from gpt_oss.torch.vision_projector import build_vision_projector
from gpt_oss.vision.image_processor import create_processor_for_vision_tower

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class MultimodalImageAnalyzer:
    """Interactive multimodal image analyzer with trained projector."""
    
    def __init__(self, checkpoint_dir="./checkpoints/poc_stage1"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = checkpoint_dir
        
        # Load trained components
        self._load_model()
        
    def _load_model(self):
        """Load the trained multimodal model components."""
        
        print("🤖 Loading trained multimodal model...")
        
        # Create config (same as training)
        self.config = ModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=4,
            num_attention_heads=4,
            mm_vision_tower="openai/clip-vit-base-patch32",
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=768,
            use_mm_proj=True,
        )
        
        # Load vision tower
        self.vision_tower = build_vision_tower(self.config)
        self.vision_tower.to(self.device)
        self.vision_tower.eval()
        
        # Load trained projector
        self.projector = build_vision_projector(self.config)
        projector_path = Path(self.checkpoint_dir) / "mm_projector.bin"
        
        if not projector_path.exists():
            raise FileNotFoundError(f"Trained projector not found: {projector_path}")
            
        state_dict = torch.load(projector_path, map_location=self.device)
        self.projector.load_state_dict(state_dict)
        self.projector.to(self.device)
        self.projector.eval()
        
        # Create image processor
        self.image_processor = create_processor_for_vision_tower(self.config.mm_vision_tower)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.device}")
        print(f"   Vision tower: {self.config.mm_vision_tower}")
        print(f"   Feature mapping: {self.config.mm_hidden_size}D → {self.config.hidden_size}D")
        
    def analyze_image(self, image_path, show_details=True):
        """Analyze a single image through the complete pipeline."""
        
        if show_details:
            print(f"\n🔍 Analyzing: {image_path}")
            print("-" * 50)
        
        try:
            # Load and display basic image info
            pil_image = Image.open(image_path)
            print(f"📷 Image: {pil_image.size} pixels, {pil_image.mode} mode")
            
            with torch.no_grad():
                # Step 1: Preprocess image
                image_tensor = self.image_processor.process(image_path)
                image_tensor = image_tensor.to(self.device)
                
                if show_details:
                    print(f"🔧 Preprocessed: {image_tensor.shape}")
                
                # Step 2: Extract vision features with CLIP
                vision_features = self.vision_tower(image_tensor)
                
                if show_details:
                    print(f"👁️  CLIP features: {vision_features.shape}")
                    print(f"   Mean: {vision_features.mean().item():.4f}")
                    print(f"   Std:  {vision_features.std().item():.4f}")
                
                # Step 3: Project to language space (THE TRAINED PART!)
                language_features = self.projector(vision_features)
                
                if show_details:
                    print(f"🧠 Language features: {language_features.shape}")
                    print(f"   Mean: {language_features.mean().item():.4f}")
                    print(f"   Std:  {language_features.std().item():.4f}")
                    print(f"   Range: [{language_features.min().item():.4f}, {language_features.max().item():.4f}]")
                
                # Step 4: Analyze the "understanding"
                self._analyze_understanding(vision_features, language_features, show_details)
                
                return {
                    'vision_features': vision_features,
                    'language_features': language_features,
                    'image_tensor': image_tensor,
                    'image_info': {
                        'size': pil_image.size,
                        'mode': pil_image.mode,
                        'path': image_path
                    }
                }
                
        except Exception as e:
            print(f"❌ Error analyzing {image_path}: {e}")
            return None
    
    def _analyze_understanding(self, vision_features, language_features, show_details):
        """Analyze what the model 'understands' about the image."""
        
        if not show_details:
            return
            
        print(f"\n🎯 Model 'Understanding' Analysis:")
        
        # Analyze spatial attention (which parts of image are important)
        # vision_features is [1, 49, 768] for 7x7 patches
        patch_importance = vision_features.norm(dim=2).squeeze(0)  # [49]
        most_important_patch = patch_importance.argmax().item()
        
        # Convert patch index to 2D coordinates (7x7 grid)
        patch_y = most_important_patch // 7
        patch_x = most_important_patch % 7
        
        print(f"   🎯 Focus area: patch ({patch_x}, {patch_y}) in 7x7 grid")
        print(f"   📊 Attention spread: {patch_importance.std().item():.4f}")
        
        # Analyze feature diversity
        feature_diversity = language_features.std(dim=1).mean().item()
        print(f"   🌈 Feature diversity: {feature_diversity:.4f}")
        
        # Analyze feature magnitude (how "confident" the model is)
        feature_magnitude = language_features.norm().item()
        print(f"   💪 Feature strength: {feature_magnitude:.2f}")
        
        # Check for activation patterns
        positive_ratio = (language_features > 0).float().mean().item()
        print(f"   ⚡ Activation ratio: {positive_ratio:.2f} (positive/negative balance)")
        
        if feature_diversity > 0.08:
            print("   ✅ Rich feature representation - model sees complex patterns")
        else:
            print("   ⚠️  Simple feature representation - basic patterns detected")
    
    def compare_images(self, image_paths):
        """Compare multiple images to see how differently the model sees them."""
        
        print(f"\n🔬 Comparing {len(image_paths)} images...")
        print("=" * 60)
        
        results = []
        for i, image_path in enumerate(image_paths):
            print(f"\n--- Image {i+1}: {Path(image_path).name} ---")
            result = self.analyze_image(image_path, show_details=False)
            if result:
                results.append(result)
        
        if len(results) < 2:
            print("Need at least 2 valid images for comparison")
            return
        
        print(f"\n📊 Cross-Image Comparison:")
        print("-" * 30)
        
        # Compare feature similarity
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                feat1 = results[i]['language_features'].flatten()
                feat2 = results[j]['language_features'].flatten()
                
                # Cosine similarity
                similarity = torch.nn.functional.cosine_similarity(feat1, feat2, dim=0).item()
                
                name1 = Path(results[i]['image_info']['path']).name
                name2 = Path(results[j]['image_info']['path']).name
                
                print(f"   {name1} ↔ {name2}: {similarity:.4f}")
                
                if similarity > 0.95:
                    print("     🟡 Very similar - model sees them as nearly identical")
                elif similarity > 0.85:
                    print("     🟢 Similar - model detects related patterns")
                elif similarity > 0.70:
                    print("     🔵 Somewhat similar - some shared features")
                else:
                    print("     🟣 Different - model sees distinct patterns")
    
    def interactive_mode(self):
        """Run in interactive mode."""
        
        print(f"\n🎮 Interactive Multimodal Demo")
        print("=" * 40)
        print("Enter image paths to analyze (or 'quit' to exit)")
        print("Examples:")
        print("  - data/poc_training/images/synthetic_0001.jpg")
        print("  - /path/to/your/image.jpg")
        print("  - compare: path1.jpg path2.jpg path3.jpg")
        
        while True:
            try:
                user_input = input(f"\n📷 Image path(s): ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower().startswith('compare:'):
                    # Compare mode
                    paths = user_input[8:].strip().split()
                    if len(paths) >= 2:
                        # Validate paths
                        valid_paths = [p for p in paths if Path(p).exists()]
                        if len(valid_paths) >= 2:
                            self.compare_images(valid_paths)
                        else:
                            print("❌ Need at least 2 valid image paths for comparison")
                    else:
                        print("❌ Usage: compare: image1.jpg image2.jpg [image3.jpg ...]")
                
                elif user_input:
                    # Single image mode
                    if Path(user_input).exists():
                        self.analyze_image(user_input)
                    else:
                        print(f"❌ Image not found: {user_input}")
                
            except KeyboardInterrupt:
                print(f"\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    """Main function."""
    
    parser = argparse.ArgumentParser(description="Interactive Multimodal Demo")
    parser.add_argument('image', nargs='?', help='Image path to analyze')
    parser.add_argument('--compare', nargs='+', help='Compare multiple images')
    parser.add_argument('--checkpoint', default='./checkpoints/poc_stage1', 
                       help='Checkpoint directory')
    
    args = parser.parse_args()
    
    # Check if trained model exists
    if not Path(args.checkpoint).exists():
        print("❌ Trained model not found!")
        print("Please run: python train_poc_stage1.py")
        return
    
    try:
        # Initialize analyzer
        analyzer = MultimodalImageAnalyzer(args.checkpoint)
        
        if args.compare:
            # Compare mode
            analyzer.compare_images(args.compare)
        elif args.image:
            # Single image mode
            analyzer.analyze_image(args.image)
        else:
            # Interactive mode
            analyzer.interactive_mode()
            
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        logger.exception("Full error details:")

if __name__ == "__main__":
    main()