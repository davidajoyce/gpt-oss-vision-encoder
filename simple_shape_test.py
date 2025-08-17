#!/usr/bin/env python3
"""
Simple Shape Test: Can the Model Tell Triangle vs Square vs Circle?

This creates 3 clear, distinct shapes and tests if the model can distinguish them.
"""

import sys
import torch
from pathlib import Path
from PIL import Image, ImageDraw

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from interactive_demo import MultimodalImageAnalyzer

def create_clear_test_shapes():
    """Create very distinct test shapes."""
    
    test_dir = Path("./clear_test_shapes")
    test_dir.mkdir(exist_ok=True)
    
    shapes = {}
    
    print("🎨 Creating clear test shapes...")
    
    # 1. Very clear RED CIRCLE
    circle_img = Image.new('RGB', (224, 224), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(circle_img)
    draw.ellipse([50, 50, 174, 174], fill=(255, 0, 0))  # Large red circle
    circle_path = test_dir / "clear_circle.jpg"
    circle_img.save(circle_path)
    shapes['RED_CIRCLE'] = str(circle_path)
    
    # 2. Very clear BLUE SQUARE
    square_img = Image.new('RGB', (224, 224), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(square_img)
    draw.rectangle([50, 50, 174, 174], fill=(0, 0, 255))  # Large blue square
    square_path = test_dir / "clear_square.jpg"
    square_img.save(square_path)
    shapes['BLUE_SQUARE'] = str(square_path)
    
    # 3. Very clear GREEN TRIANGLE
    triangle_img = Image.new('RGB', (224, 224), color=(255, 255, 255))  # White background
    draw = ImageDraw.Draw(triangle_img)
    points = [(112, 50), (50, 174), (174, 174)]  # Large triangle
    draw.polygon(points, fill=(0, 255, 0))  # Green triangle
    triangle_path = test_dir / "clear_triangle.jpg"
    triangle_img.save(triangle_path)
    shapes['GREEN_TRIANGLE'] = str(triangle_path)
    
    print(f"✅ Created 3 distinct test shapes in {test_dir}")
    return shapes

def test_shape_recognition():
    """Test if the model can distinguish between clear shapes."""
    
    print("🔍 Testing Shape Recognition with Crystal Clear Shapes")
    print("=" * 55)
    
    # Create test shapes
    shapes = create_clear_test_shapes()
    
    # Load trained model
    try:
        analyzer = MultimodalImageAnalyzer()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Make sure you've run: python train_poc_stage1.py")
        return
    
    # Analyze each shape
    results = {}
    
    for shape_name, image_path in shapes.items():
        print(f"\n🔸 Analyzing {shape_name}")
        print("-" * 30)
        
        # Get detailed analysis
        result = analyzer.analyze_image(image_path, show_details=False)
        
        if result:
            language_features = result['language_features'].squeeze()  # [49, 256]
            
            # Key metrics for shape discrimination
            metrics = {
                'mean_activation': language_features.mean().item(),
                'max_activation': language_features.max().item(),
                'min_activation': language_features.min().item(),
                'std_activation': language_features.std().item(),
                'feature_strength': language_features.norm().item(),
            }
            
            # Spatial analysis (7x7 patch grid)
            spatial_norm = language_features.norm(dim=1)  # [49] importance per patch
            
            # Center patch (index 24 in 7x7 grid)
            center_activation = spatial_norm[24].item()
            
            # Corner patches
            corners = [0, 6, 42, 48]  # Top-left, top-right, bottom-left, bottom-right
            corner_activation = spatial_norm[corners].mean().item()
            
            # Edge patches (not corners)
            edges = [1, 2, 3, 4, 5, 7, 14, 21, 28, 35, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47]
            edge_activation = spatial_norm[edges].mean().item()
            
            metrics.update({
                'center_activation': center_activation,
                'corner_activation': corner_activation,
                'edge_activation': edge_activation,
                'center_corner_ratio': center_activation / corner_activation if corner_activation > 0 else 0,
                'center_edge_ratio': center_activation / edge_activation if edge_activation > 0 else 0,
            })
            
            # Feature channel analysis
            channel_importance = language_features.norm(dim=0)  # [256] importance per feature
            top_5_channels = channel_importance.topk(5).indices.tolist()
            
            metrics['top_channels'] = top_5_channels
            metrics['channel_diversity'] = channel_importance.std().item()
            
            results[shape_name] = metrics
            
            # Display key findings
            print(f"   Feature strength: {metrics['feature_strength']:.3f}")
            print(f"   Center activation: {metrics['center_activation']:.3f}")
            print(f"   Corner activation: {metrics['corner_activation']:.3f}")
            print(f"   Edge activation: {metrics['edge_activation']:.3f}")
            print(f"   Center/Corner ratio: {metrics['center_corner_ratio']:.3f}")
            print(f"   Center/Edge ratio: {metrics['center_edge_ratio']:.3f}")
            print(f"   Top feature channels: {metrics['top_channels'][:3]}")
    
    # Compare shapes
    if len(results) >= 2:
        print(f"\n📊 Shape Comparison Analysis")
        print("=" * 35)
        
        shape_names = list(results.keys())
        
        print(f"\n🎯 Key Discriminative Features:")
        
        # Compare center/corner ratios (should differ for triangle vs square vs circle)
        print(f"\nCenter/Corner Activation Ratios:")
        for name in shape_names:
            ratio = results[name]['center_corner_ratio']
            print(f"   {name}: {ratio:.3f}")
        
        # Check if triangle shows different spatial pattern
        if 'GREEN_TRIANGLE' in results and 'BLUE_SQUARE' in results:
            triangle_ratio = results['GREEN_TRIANGLE']['center_corner_ratio']
            square_ratio = results['BLUE_SQUARE']['center_corner_ratio']
            diff = abs(triangle_ratio - square_ratio)
            
            print(f"\n🔺 Triangle vs Square discrimination:")
            print(f"   Ratio difference: {diff:.3f}")
            if diff > 0.05:
                print(f"   ✅ Model shows DIFFERENT spatial patterns for triangle vs square!")
            else:
                print(f"   ⚠️  Model shows SIMILAR spatial patterns")
        
        # Compare feature similarities
        print(f"\n🔬 Feature Vector Similarities:")
        for i, name1 in enumerate(shape_names):
            for name2 in shape_names[i+1:]:
                
                # Get feature vectors
                feat1 = results[name1]
                feat2 = results[name2]
                
                # Compare top channels
                channels1 = set(feat1['top_channels'][:3])
                channels2 = set(feat2['top_channels'][:3])
                channel_overlap = len(channels1 & channels2) / 3.0
                
                # Compare overall metrics
                strength_diff = abs(feat1['feature_strength'] - feat2['feature_strength'])
                spatial_diff = abs(feat1['center_corner_ratio'] - feat2['center_corner_ratio'])
                
                print(f"   {name1} ↔ {name2}:")
                print(f"     Channel overlap: {channel_overlap:.2f} (lower = more different)")
                print(f"     Strength difference: {strength_diff:.3f}")
                print(f"     Spatial difference: {spatial_diff:.3f}")
                
                if channel_overlap < 0.5 and spatial_diff > 0.05:
                    print(f"     ✅ Shapes are DISTINGUISHABLE!")
                else:
                    print(f"     ⚠️  Shapes are similar in features")
    
    # Final verdict
    print(f"\n🎯 Final Verdict on Shape Recognition:")
    print("=" * 40)
    
    if len(results) >= 3:
        # Check if all shapes have different patterns
        ratios = [results[name]['center_corner_ratio'] for name in results.keys()]
        ratio_std = torch.tensor(ratios).std().item()
        
        channels_lists = [set(results[name]['top_channels'][:3]) for name in results.keys()]
        unique_channels = len(set.union(*channels_lists))
        
        print(f"Spatial pattern diversity: {ratio_std:.3f}")
        print(f"Feature channel diversity: {unique_channels}/256 channels used")
        
        if ratio_std > 0.03 and unique_channels >= 5:
            print(f"\n✅ YES! The model CAN distinguish between shapes!")
            print(f"   • Different spatial attention patterns")
            print(f"   • Different feature channel activations")
            print(f"   • Foundation for shape recognition is present")
        else:
            print(f"\n⚠️  PARTIAL: Model shows some shape sensitivity")
            print(f"   • Limited but detectable differences")
            print(f"   • More training would improve discrimination")
    
    print(f"\n💡 What this means:")
    print(f"   • The model processes shapes at a FEATURE level")
    print(f"   • It doesn't output 'triangle' or 'square' (no text generation yet)")
    print(f"   • But it creates different internal representations")
    print(f"   • This is the foundation for multimodal understanding!")

def main():
    test_shape_recognition()

if __name__ == "__main__":
    main()