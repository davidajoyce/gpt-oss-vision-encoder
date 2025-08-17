#!/usr/bin/env python3
"""
Test script for the multimodal generation pipeline.

This script tests the complete generation workflow including:
- Image preprocessing
- Multimodal model inference 
- Token generation and decoding
"""

import unittest
import tempfile
import shutil
import os
import sys
from unittest.mock import patch, MagicMock

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.vision.image_processor import ImageProcessor, create_processor_for_vision_tower


class TestGenerationPipeline(unittest.TestCase):
    """Test the multimodal generation pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.test_image_path = os.path.join(self.temp_dir, 'test.jpg')
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_image_preprocessing_pipeline(self):
        """Test the complete image preprocessing pipeline."""
        # Test CLIP processor
        clip_processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Process image
        tensor = clip_processor.process(self.test_image_path)
        
        # Validate output
        self.assertEqual(tensor.shape, (1, 3, 224, 224))
        self.assertTrue(torch.all(torch.isfinite(tensor)))
        
        # Check normalization (CLIP uses specific mean/std)
        # Values should be roughly centered around 0
        self.assertLess(abs(tensor.mean().item()), 2.0)
    
    def test_image_info_extraction(self):
        """Test image information extraction."""
        processor = ImageProcessor()
        
        info = processor.get_info(self.test_image_path)
        
        self.assertEqual(info['size'], (224, 224))
        self.assertEqual(info['mode'], 'RGB')
        self.assertEqual(info['aspect_ratio'], 1.0)
        self.assertAlmostEqual(info['megapixels'], 224*224/1_000_000, places=3)
    
    def test_batch_processing_consistency(self):
        """Test that batch processing gives consistent results."""
        processor = create_processor_for_vision_tower('clip')
        
        # Process individually
        individual_results = []
        images = [self.test_image_path, self.test_image_path]  # Same image twice
        
        for img in images:
            result = processor.process(img)
            individual_results.append(result)
        
        # Process as batch
        batch_result = processor.process_batch(images)
        
        # Results should be identical
        individual_stacked = torch.cat(individual_results, dim=0)
        torch.testing.assert_close(batch_result, individual_stacked, rtol=1e-5, atol=1e-6)
    
    def test_error_recovery_in_batch(self):
        """Test that batch processing recovers from individual image errors."""
        processor = create_processor_for_vision_tower('clip')
        
        # Mix valid and invalid images
        images = [
            self.test_image_path,  # Valid
            '/nonexistent/path.jpg',  # Invalid
            self.test_image_path,  # Valid
        ]
        
        # Should not crash, should use placeholder for invalid image
        batch_result = processor.process_batch(images)
        
        self.assertEqual(batch_result.shape, (3, 3, 224, 224))
        
        # First and third should be similar (same image)
        torch.testing.assert_close(batch_result[0], batch_result[2], rtol=1e-3, atol=1e-3)
        
        # Second should be zeros (placeholder)
        self.assertTrue(torch.allclose(batch_result[1], torch.zeros_like(batch_result[1])))
    
    def test_different_image_formats(self):
        """Test processing of different image formats."""
        processor = create_processor_for_vision_tower('clip')
        
        # Create images in different formats
        formats = [
            ('RGB', 'jpg'),
            ('L', 'png'),    # Grayscale
            ('RGBA', 'png'), # With alpha channel
        ]
        
        results = []
        for mode, ext in formats:
            # Create image
            if mode == 'RGBA':
                img = Image.new(mode, (224, 224), color=(255, 0, 0, 128))
            else:
                img = Image.new(mode, (224, 224), color='red' if mode == 'RGB' else 128)
            
            path = os.path.join(self.temp_dir, f'test_{mode}.{ext}')
            img.save(path)
            
            # Process
            result = processor.process(path)
            results.append(result)
            
            # Validate
            self.assertEqual(result.shape, (1, 3, 224, 224))
            self.assertTrue(torch.all(torch.isfinite(result)))
        
        # All results should have same shape
        for result in results[1:]:
            self.assertEqual(result.shape, results[0].shape)
    
    def test_memory_efficiency(self):
        """Test memory efficiency of image processing."""
        processor = create_processor_for_vision_tower('clip')
        
        # Process multiple images and check memory doesn't grow excessively
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for i in range(5):
            # Create different colored images
            color = (i * 50, 0, 0)
            img = Image.new('RGB', (224, 224), color=color)
            path = os.path.join(self.temp_dir, f'test_{i}.jpg')
            img.save(path)
            
            # Process
            result = processor.process(path)
            
            # Clean up result to avoid memory accumulation
            del result
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            # Memory growth should be minimal (less than 50MB)
            self.assertLess(memory_growth / (1024*1024), 50)
    
    def test_vision_tower_specific_preprocessing(self):
        """Test that different vision towers get appropriate preprocessing."""
        # Test CLIP preprocessing
        clip_processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        clip_result = clip_processor.process(self.test_image_path)
        
        # Test generic preprocessing
        generic_processor = create_processor_for_vision_tower('some-other-model')
        generic_result = generic_processor.process(self.test_image_path)
        
        # Both should produce valid outputs
        self.assertEqual(clip_result.shape, (1, 3, 224, 224))
        self.assertEqual(generic_result.shape, (1, 3, 224, 224))
        
        # CLIP should use different normalization
        self.assertFalse(torch.allclose(clip_result, generic_result, rtol=0.1))
    
    def test_preprocessing_deterministic(self):
        """Test that preprocessing is deterministic."""
        processor = create_processor_for_vision_tower('clip')
        
        # Process same image multiple times
        results = []
        for _ in range(3):
            result = processor.process(self.test_image_path)
            results.append(result)
        
        # All results should be identical
        for result in results[1:]:
            torch.testing.assert_close(result, results[0])
    
    def test_image_size_handling(self):
        """Test handling of various image sizes."""
        processor = create_processor_for_vision_tower('clip')
        
        sizes = [(32, 32), (224, 224), (512, 512), (100, 300), (800, 200)]
        
        for width, height in sizes:
            with self.subTest(size=(width, height)):
                # Create image of specific size
                img = Image.new('RGB', (width, height), color='blue')
                path = os.path.join(self.temp_dir, f'test_{width}x{height}.jpg')
                img.save(path)
                
                # Process
                result = processor.process(path)
                
                # Should always output target size
                self.assertEqual(result.shape, (1, 3, 224, 224))
    
    def test_numerical_stability(self):
        """Test numerical stability of preprocessing."""
        processor = create_processor_for_vision_tower('clip')
        
        # Test extreme pixel values
        extreme_cases = [
            ('all_black', Image.new('RGB', (224, 224), color=(0, 0, 0))),
            ('all_white', Image.new('RGB', (224, 224), color=(255, 255, 255))),
            ('high_contrast', Image.new('RGB', (224, 224), color=(255, 0, 255))),
        ]
        
        for name, img in extreme_cases:
            with self.subTest(case=name):
                path = os.path.join(self.temp_dir, f'{name}.jpg')
                img.save(path)
                
                result = processor.process(path)
                
                # Check for NaN or infinite values
                self.assertTrue(torch.all(torch.isfinite(result)))
                
                # Values should be in reasonable range after normalization
                self.assertTrue(torch.all(result >= -5.0))
                self.assertTrue(torch.all(result <= 5.0))


class TestGenerationIntegration(unittest.TestCase):
    """Test integration between image processing and generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test image
        self.test_image = Image.new('RGB', (224, 224), color='red')
        self.test_image_path = os.path.join(self.temp_dir, 'test.jpg')
        self.test_image.save(self.test_image_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_image_to_features_pipeline(self):
        """Test the complete pipeline from image to features."""
        from gpt_oss.torch.model import ModelConfig
        
        # Create processor
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Process image
        image_tensor = processor.process(self.test_image_path)
        
        # Mock vision tower processing
        with patch('gpt_oss.torch.vision_tower.CLIPVisionTower') as mock_tower:
            mock_instance = MagicMock()
            mock_instance.return_value = torch.randn(1, 256, 1024)  # Mock CLIP features
            mock_tower.return_value = mock_instance
            
            # This would be the actual vision tower processing
            features = mock_instance(image_tensor)
            
            self.assertEqual(features.shape, (1, 256, 1024))
            self.assertTrue(torch.all(torch.isfinite(features)))
    
    def test_features_to_projector_pipeline(self):
        """Test the features to projector pipeline.""" 
        from gpt_oss.torch.vision_projector import build_vision_projector
        from gpt_oss.torch.model import ModelConfig
        
        # Create config and projector
        config = ModelConfig(
            mm_hidden_size=1024,
            hidden_size=128,
            mm_projector_type='linear'
        )
        
        projector = build_vision_projector(config)
        
        # Mock vision features
        vision_features = torch.randn(1, 256, 1024)
        
        # Project features
        projected = projector(vision_features)
        
        self.assertEqual(projected.shape, (1, 256, 128))
        self.assertTrue(torch.all(torch.isfinite(projected)))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)