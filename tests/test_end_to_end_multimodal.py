#!/usr/bin/env python3
"""
End-to-end tests for multimodal inference pipeline.

These tests validate the complete multimodal pipeline from image input 
to text generation, ensuring all components work together correctly.
"""

import unittest
import tempfile
import shutil
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
import logging

import torch
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.generate_multimodal import MultimodalTokenGenerator
from gpt_oss.vision.image_processor import ImageProcessor, create_processor_for_vision_tower


class TestEndToEndMultimodal(unittest.TestCase):
    """End-to-end tests for the complete multimodal pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')  # Use CPU for testing
        
        # Create a minimal multimodal model config
        self.config = ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            # Vision parameters
            mm_vision_tower="openai/clip-vit-base-patch32",
            mm_projector_type="linear",
            mm_hidden_size=512,
            use_mm_proj=True,
        )
        
        # Create and save a test model
        self.model = Transformer(self.config, device=self.device)
        self.checkpoint_path = os.path.join(self.temp_dir, 'test_model.safetensors')
        
        # Create test images
        self.test_images = self._create_test_images()
        
        # Mock tokenizer
        self.mock_tokenizer = self._create_mock_tokenizer()
        
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_images(self):
        """Create test images in various formats."""
        images = {}
        
        # RGB image
        rgb_image = Image.new('RGB', (224, 224), color='red')
        rgb_path = os.path.join(self.temp_dir, 'test_rgb.jpg')
        rgb_image.save(rgb_path)
        images['rgb_path'] = rgb_path
        images['rgb_pil'] = rgb_image
        
        # Numpy array
        rgb_array = np.array(rgb_image)
        images['rgb_array'] = rgb_array
        
        # Grayscale image
        gray_image = Image.new('L', (224, 224), color=128)
        gray_path = os.path.join(self.temp_dir, 'test_gray.png')
        gray_image.save(gray_path)
        images['gray_path'] = gray_path
        
        # Small image
        small_image = Image.new('RGB', (64, 64), color='blue')
        small_path = os.path.join(self.temp_dir, 'test_small.jpg')
        small_image.save(small_path)
        images['small_path'] = small_path
        
        # Large image
        large_image = Image.new('RGB', (1024, 1024), color='green')
        large_path = os.path.join(self.temp_dir, 'test_large.jpg')
        large_image.save(large_path)
        images['large_path'] = large_path
        
        return images
    
    def _create_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        class MockTokenizer:
            def __init__(self):
                self.eot_token = 999
                
            def encode(self, text):
                # Simple mock encoding: split words and hash
                words = text.split()
                return [hash(word) % 500 + 1 for word in words]  # Avoid 0 and eot_token
            
            def decode(self, tokens):
                # Simple mock decoding
                if isinstance(tokens, list) and len(tokens) == 1:
                    return f"token_{tokens[0]}"
                return f"decoded_{len(tokens)}_tokens"
        
        return MockTokenizer()
    
    def test_image_processor_handles_various_formats(self):
        """Test that ImageProcessor handles various image formats correctly."""
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Test different input formats
        test_cases = [
            ('RGB PIL Image', self.test_images['rgb_pil']),
            ('RGB file path', self.test_images['rgb_path']),
            ('RGB numpy array', self.test_images['rgb_array']),
            ('Grayscale file path', self.test_images['gray_path']),
            ('Small image', self.test_images['small_path']),
            ('Large image', self.test_images['large_path']),
        ]
        
        for description, image_input in test_cases:
            with self.subTest(description=description):
                try:
                    result = processor.process(image_input)
                    
                    # Validate output tensor
                    self.assertIsInstance(result, torch.Tensor)
                    self.assertEqual(result.shape, (1, 3, 224, 224))  # [B, C, H, W]
                    self.assertTrue(torch.all(torch.isfinite(result)))
                    
                except Exception as e:
                    self.fail(f"Failed to process {description}: {e}")
    
    def test_multimodal_generator_initialization(self):
        """Test MultimodalTokenGenerator initialization."""
        with patch('gpt_oss.generate_multimodal.get_tokenizer', return_value=self.mock_tokenizer):
            # Test text-only initialization
            generator = MultimodalTokenGenerator(
                checkpoint=self.checkpoint_path,
                device=self.device
            )
            self.assertFalse(generator.multimodal_enabled)
            
            # Test multimodal initialization
            with patch('gpt_oss.torch.vision_tower.build_vision_tower') as mock_vision:
                with patch('gpt_oss.torch.vision_projector.build_vision_projector') as mock_projector:
                    mock_vision.return_value = MagicMock()
                    mock_projector.return_value = MagicMock()
                    
                    generator = MultimodalTokenGenerator(
                        checkpoint=self.checkpoint_path,
                        device=self.device,
                        vision_tower='openai/clip-vit-base-patch32'
                    )
                    self.assertTrue(generator.multimodal_enabled)
    
    def test_text_only_generation_backward_compatibility(self):
        """Test that text-only generation still works (backward compatibility)."""
        with patch('gpt_oss.generate_multimodal.get_tokenizer', return_value=self.mock_tokenizer):
            with patch.object(Transformer, 'from_checkpoint', return_value=self.model):
                generator = MultimodalTokenGenerator(
                    checkpoint=self.checkpoint_path,
                    device=self.device
                )
                
                # Generate from text prompt
                prompt_tokens = [1, 2, 3]  # Mock tokens
                stop_tokens = [999]  # Mock EOT token
                
                generated_tokens = []
                for token, logprob in generator.generate(
                    prompt_tokens=prompt_tokens,
                    stop_tokens=stop_tokens,
                    max_tokens=5,
                    return_logprobs=True
                ):
                    generated_tokens.append(token)
                    self.assertIsInstance(token, int)
                    self.assertIsInstance(logprob, float)
                    
                    if len(generated_tokens) >= 3:  # Limit for testing
                        break
                
                self.assertGreater(len(generated_tokens), 0)
    
    def test_multimodal_generation_with_image(self):
        """Test multimodal generation with image input."""
        with patch('gpt_oss.generate_multimodal.get_tokenizer', return_value=self.mock_tokenizer):
            with patch.object(Transformer, 'from_checkpoint', return_value=self.model):
                with patch('gpt_oss.torch.vision_tower.build_vision_tower') as mock_vision:
                    with patch('gpt_oss.torch.vision_projector.build_vision_projector') as mock_projector:
                        # Set up mocks
                        mock_vision_tower = MagicMock()
                        mock_vision_tower.return_value = torch.randn(1, 256, 512)  # Mock vision features
                        mock_vision.return_value = mock_vision_tower
                        
                        mock_proj = MagicMock()
                        mock_proj.return_value = torch.randn(1, 256, 128)  # Mock projected features
                        mock_projector.return_value = mock_proj
                        
                        generator = MultimodalTokenGenerator(
                            checkpoint=self.checkpoint_path,
                            device=self.device,
                            vision_tower='openai/clip-vit-base-patch32'
                        )
                        
                        # Generate from text + image
                        prompt_tokens = [1, 2, 3]
                        stop_tokens = [999]
                        
                        generated_tokens = []
                        for token, logprob in generator.generate(
                            prompt_tokens=prompt_tokens,
                            stop_tokens=stop_tokens,
                            image=self.test_images['rgb_path'],
                            max_tokens=3,
                            return_logprobs=True
                        ):
                            generated_tokens.append(token)
                            self.assertIsInstance(token, int)
                            self.assertIsInstance(logprob, float)
                        
                        self.assertGreater(len(generated_tokens), 0)
                        
                        # Verify vision components were called
                        mock_vision_tower.assert_called()
                        mock_proj.assert_called()
    
    def test_error_handling_invalid_image(self):
        """Test error handling for invalid image inputs."""
        with patch('gpt_oss.generate_multimodal.get_tokenizer', return_value=self.mock_tokenizer):
            with patch.object(Transformer, 'from_checkpoint', return_value=self.model):
                with patch('gpt_oss.torch.vision_tower.build_vision_tower'):
                    with patch('gpt_oss.torch.vision_projector.build_vision_projector'):
                        generator = MultimodalTokenGenerator(
                            checkpoint=self.checkpoint_path,
                            device=self.device,
                            vision_tower='openai/clip-vit-base-patch32'
                        )
                        
                        # Test with non-existent image file
                        with self.assertRaises(Exception):
                            generator.process_image('/nonexistent/path.jpg')
                        
                        # Test with corrupted data
                        with self.assertRaises(Exception):
                            generator.process_image(b'invalid_image_data')
    
    def test_image_processing_pipeline_consistency(self):
        """Test that image processing is consistent across different inputs."""
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Process the same image in different formats
        rgb_path_result = processor.process(self.test_images['rgb_path'])
        rgb_pil_result = processor.process(self.test_images['rgb_pil'])
        rgb_array_result = processor.process(self.test_images['rgb_array'])
        
        # Results should be very similar (allowing for minor encoding differences)
        self.assertEqual(rgb_path_result.shape, rgb_pil_result.shape)
        self.assertEqual(rgb_pil_result.shape, rgb_array_result.shape)
        
        # Check that values are in reasonable range
        for result in [rgb_path_result, rgb_pil_result, rgb_array_result]:
            self.assertTrue(torch.all(result >= -3.0))  # Reasonable lower bound after normalization
            self.assertTrue(torch.all(result <= 3.0))   # Reasonable upper bound after normalization
    
    def test_batch_image_processing(self):
        """Test batch processing of multiple images."""
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Create batch of different images
        image_batch = [
            self.test_images['rgb_path'],
            self.test_images['gray_path'],
            self.test_images['small_path'],
        ]
        
        # Process batch
        batch_result = processor.process_batch(image_batch)
        
        # Validate batch output
        self.assertEqual(batch_result.shape, (3, 3, 224, 224))  # [B, C, H, W]
        self.assertTrue(torch.all(torch.isfinite(batch_result)))
    
    def test_memory_usage_reasonable(self):
        """Test that memory usage is reasonable for typical inputs."""
        processor = create_processor_for_vision_tower('openai/clip-vit-base-patch32')
        
        # Track memory before processing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Process a typical image
        result = processor.process(self.test_images['rgb_path'])
        
        # Check memory usage is reasonable
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated()
            memory_used_mb = (final_memory - initial_memory) / (1024 * 1024)
            self.assertLess(memory_used_mb, 100, "Memory usage should be less than 100MB for single image")
        
        # Check output size is reasonable  
        output_size_mb = result.numel() * result.element_size() / (1024 * 1024)
        self.assertLess(output_size_mb, 10, "Output tensor should be less than 10MB")
    
    def test_generation_speed_reasonable(self):
        """Test that generation speed is reasonable."""
        import time
        
        with patch('gpt_oss.generate_multimodal.get_tokenizer', return_value=self.mock_tokenizer):
            with patch.object(Transformer, 'from_checkpoint', return_value=self.model):
                generator = MultimodalTokenGenerator(
                    checkpoint=self.checkpoint_path,
                    device=self.device
                )
                
                # Time text-only generation
                start_time = time.time()
                tokens_generated = 0
                for token, _ in generator.generate(
                    prompt_tokens=[1, 2, 3],
                    stop_tokens=[999],
                    max_tokens=10
                ):
                    tokens_generated += 1
                text_time = time.time() - start_time
                
                # Should generate at reasonable speed (>1 token/second even on CPU)
                if tokens_generated > 0:
                    tokens_per_second = tokens_generated / text_time
                    self.assertGreater(tokens_per_second, 0.5, 
                                     f"Generation too slow: {tokens_per_second:.2f} tokens/sec")


class TestImageProcessorEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions for image processing."""
    
    def setUp(self):
        self.processor = ImageProcessor()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_very_small_images(self):
        """Test processing of very small images."""
        # Create 1x1 pixel image
        tiny_image = Image.new('RGB', (1, 1), color='red')
        result = self.processor.process(tiny_image)
        
        # Should still produce correct output shape
        self.assertEqual(result.shape, (1, 3, 224, 224))
    
    def test_very_large_images(self):
        """Test processing of large images."""
        # Create large image (simulate without actually creating huge file)
        with patch.object(Image, 'open') as mock_open:
            large_image = Image.new('RGB', (4096, 4096), color='blue')
            mock_open.return_value = large_image
            
            # Should handle large images without error
            result = self.processor.process('fake_large_image.jpg')
            self.assertEqual(result.shape, (1, 3, 224, 224))
    
    def test_extreme_aspect_ratios(self):
        """Test images with extreme aspect ratios."""
        # Very wide image
        wide_image = Image.new('RGB', (1000, 10), color='red')
        result = self.processor.process(wide_image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        
        # Very tall image
        tall_image = Image.new('RGB', (10, 1000), color='blue')
        result = self.processor.process(tall_image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
    
    def test_corrupted_file_handling(self):
        """Test handling of corrupted image files."""
        # Create file with invalid image data
        corrupted_path = os.path.join(self.temp_dir, 'corrupted.jpg')
        with open(corrupted_path, 'wb') as f:
            f.write(b'not_an_image')
        
        # Should raise appropriate error
        with self.assertRaises(ValueError):
            self.processor.process(corrupted_path)
    
    def test_unsupported_formats(self):
        """Test handling of unsupported file formats."""
        # Create file with unsupported extension
        unsupported_path = os.path.join(self.temp_dir, 'test.xyz')
        with open(unsupported_path, 'wb') as f:
            f.write(b'some data')
        
        # Should raise appropriate error
        with self.assertRaises(ValueError):
            self.processor.process(unsupported_path)


if __name__ == '__main__':
    # Configure logging for tests
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    unittest.main(verbosity=2)