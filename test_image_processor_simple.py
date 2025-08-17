#!/usr/bin/env python3
"""
Simple test script for image processor functionality.
"""

import tempfile
import shutil
import os
from PIL import Image
import torch
import numpy as np

# Add project root to path
import sys
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.vision.image_processor import ImageProcessor, create_processor_for_vision_tower


def test_basic_functionality():
    """Test basic image processor functionality."""
    print("Testing basic image processor functionality...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_path = os.path.join(temp_dir, 'test.jpg')
        test_image.save(test_path)
        
        # Create processor
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Test different input types
        print("✓ Testing file path input...")
        result1 = processor.process(test_path)
        assert result1.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {result1.shape}"
        
        print("✓ Testing PIL Image input...")
        result2 = processor.process(test_image)
        assert result2.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {result2.shape}"
        
        print("✓ Testing numpy array input...")
        image_array = np.array(test_image)
        result3 = processor.process(image_array)
        assert result3.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {result3.shape}"
        
        # Test batch processing
        print("✓ Testing batch processing...")
        batch_result = processor.process_batch([test_path, test_image, image_array])
        assert batch_result.shape == (3, 3, 224, 224), f"Expected (3, 3, 224, 224), got {batch_result.shape}"
        
        # Test image info
        print("✓ Testing image info extraction...")
        info = processor.get_info(test_path)
        assert info['size'] == (224, 224), f"Expected size (224, 224), got {info['size']}"
        assert info['mode'] == 'RGB', f"Expected mode RGB, got {info['mode']}"
        
        print("✅ All basic functionality tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        processor = create_processor_for_vision_tower('clip')
        
        # Test different image sizes
        print("✓ Testing various image sizes...")
        sizes = [(32, 32), (512, 512), (100, 300)]
        for width, height in sizes:
            img = Image.new('RGB', (width, height), color='blue')
            path = os.path.join(temp_dir, f'test_{width}x{height}.jpg')
            img.save(path)
            
            result = processor.process(path)
            assert result.shape == (1, 3, 224, 224), f"Size {width}x{height} failed: {result.shape}"
        
        # Test different modes
        print("✓ Testing different image modes...")
        modes = [('RGB', 'red'), ('L', 128)]  # RGB and Grayscale
        for mode, color in modes:
            img = Image.new(mode, (224, 224), color=color)
            path = os.path.join(temp_dir, f'test_{mode}.png')
            img.save(path)
            
            result = processor.process(path)
            assert result.shape == (1, 3, 224, 224), f"Mode {mode} failed: {result.shape}"
        
        # Test error handling
        print("✓ Testing error handling...")
        try:
            processor.process('/nonexistent/path.jpg')
            assert False, "Should have raised an error for non-existent file"
        except (FileNotFoundError, ValueError):
            pass  # Expected
        
        print("✅ All edge case tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_vision_tower_specific():
    """Test vision tower specific processing."""
    print("\nTesting vision tower specific processing...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        # Create test image
        test_image = Image.new('RGB', (224, 224), color='green')
        test_path = os.path.join(temp_dir, 'test.jpg')
        test_image.save(test_path)
        
        # Test CLIP processor
        print("✓ Testing CLIP processor...")
        clip_processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        clip_result = clip_processor.process(test_path)
        
        # Test generic processor
        print("✓ Testing generic processor...")
        generic_processor = create_processor_for_vision_tower('some-other-model')
        generic_result = generic_processor.process(test_path)
        
        # Both should work but produce different results (different normalization)
        assert clip_result.shape == generic_result.shape
        assert not torch.allclose(clip_result, generic_result, rtol=0.1), "Results should differ due to different normalization"
        
        print("✅ Vision tower specific tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)


def test_numerical_properties():
    """Test numerical properties of processed images."""
    print("\nTesting numerical properties...")
    
    temp_dir = tempfile.mkdtemp()
    try:
        processor = create_processor_for_vision_tower('clip')
        
        # Test extreme cases
        extreme_cases = [
            ('black', (0, 0, 0)),
            ('white', (255, 255, 255)),
            ('red', (255, 0, 0)),
        ]
        
        for name, color in extreme_cases:
            print(f"✓ Testing {name} image...")
            img = Image.new('RGB', (224, 224), color=color)
            path = os.path.join(temp_dir, f'{name}.jpg')
            img.save(path)
            
            result = processor.process(path)
            
            # Check for numerical stability
            assert torch.all(torch.isfinite(result)), f"{name} image produced non-finite values"
            assert torch.all(result >= -10), f"{name} image produced values too low: {result.min()}"
            assert torch.all(result <= 10), f"{name} image produced values too high: {result.max()}"
        
        # Test deterministic processing
        print("✓ Testing deterministic processing...")
        img = Image.new('RGB', (224, 224), color='blue')
        path = os.path.join(temp_dir, 'deterministic.jpg')
        img.save(path)
        
        results = [processor.process(path) for _ in range(3)]
        for i, result in enumerate(results[1:], 1):
            assert torch.allclose(result, results[0]), f"Processing not deterministic at iteration {i}"
        
        print("✅ All numerical property tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    print("🚀 Starting Image Processor Tests\n")
    
    try:
        test_basic_functionality()
        test_edge_cases()
        test_vision_tower_specific()
        test_numerical_properties()
        
        print("\n🎉 All tests passed! Image processor is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise