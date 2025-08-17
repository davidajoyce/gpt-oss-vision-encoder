#!/usr/bin/env python3
"""
Test Your Own Image!

Quick script to test any image with the trained multimodal model.
Shows how the model "sees" and "understands" your image at the feature level.

Usage:
    python test_your_image.py path/to/your/image.jpg
    
    # Or drag and drop an image path
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from interactive_demo import MultimodalImageAnalyzer

def main():
    if len(sys.argv) != 2:
        print("🖼️  Test Your Image with Trained Multimodal Model")
        print("=" * 50)
        print("Usage: python test_your_image.py path/to/image.jpg")
        print("\nOr try a synthetic image:")
        print("python test_your_image.py data/poc_training/images/synthetic_0042.jpg")
        return
    
    image_path = sys.argv[1]
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return
    
    print("🎯 Testing Your Image with Trained Multimodal Model")
    print("=" * 55)
    
    try:
        # Load the trained model
        analyzer = MultimodalImageAnalyzer()
        
        # Analyze the image
        result = analyzer.analyze_image(image_path)
        
        if result:
            print("\n✅ Analysis complete!")
            print("\n🎯 What this shows:")
            print("   • CLIP extracted visual features from your image")
            print("   • Our trained projector mapped them to language space")
            print("   • The model has created a feature representation")
            print("   • This is the foundation for vision-language understanding!")
            
            print("\n🚀 For full multimodal chat:")
            print("   • Train Stage 2 with instruction-following data")
            print("   • Add text generation on top of these features")
            print("   • Scale to real datasets and larger models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nMake sure you've run the training first:")
        print("python train_poc_stage1.py")

if __name__ == "__main__":
    main()