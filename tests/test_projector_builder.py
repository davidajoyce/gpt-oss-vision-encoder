import sys
import os
import unittest

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from gpt_oss.torch.model import ModelConfig
from gpt_oss.torch.vision_projector import build_vision_projector, IdentityMap, SimpleResBlock


class TestProjectorBuilder(unittest.TestCase):
    """Test vision projector builder functionality."""

    def test_linear_projector(self):
        """Test linear projector creation."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="linear"
        )
        
        projector = build_vision_projector(config)
        
        self.assertIsInstance(projector, nn.Linear)
        self.assertEqual(projector.in_features, 768)
        self.assertEqual(projector.out_features, 2048)

    def test_mlp_projector(self):
        """Test MLP projector creation."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="mlp2x_gelu"
        )
        
        projector = build_vision_projector(config)
        
        self.assertIsInstance(projector, nn.Sequential)
        # Should have 3 modules: Linear -> GELU -> Linear
        self.assertEqual(len(projector), 3)
        self.assertIsInstance(projector[0], nn.Linear)
        self.assertIsInstance(projector[1], nn.GELU)
        self.assertIsInstance(projector[2], nn.Linear)
        
        # Check dimensions
        self.assertEqual(projector[0].in_features, 768)
        self.assertEqual(projector[0].out_features, 2048)
        self.assertEqual(projector[2].in_features, 2048)
        self.assertEqual(projector[2].out_features, 2048)

    def test_identity_projector(self):
        """Test identity projector creation."""
        config = ModelConfig(
            mm_hidden_size=2048,
            hidden_size=2048,
            mm_projector_type="identity"
        )
        
        projector = build_vision_projector(config)
        
        self.assertIsInstance(projector, IdentityMap)

    def test_identity_projector_dimension_mismatch(self):
        """Test that identity projector raises error when dimensions don't match."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="identity"
        )
        
        with self.assertRaises(ValueError):
            build_vision_projector(config)

    def test_projector_missing_mm_hidden_size(self):
        """Test that projector builder raises error when mm_hidden_size is missing."""
        config = ModelConfig(
            hidden_size=2048,
            mm_projector_type="linear"
        )
        
        with self.assertRaises(ValueError):
            build_vision_projector(config)

    def test_unknown_projector_type(self):
        """Test that unknown projector type raises error."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="unknown_type"
        )
        
        with self.assertRaises(ValueError):
            build_vision_projector(config)

    def test_projector_forward_pass(self):
        """Test that projector forward pass works correctly."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="linear"
        )
        
        projector = build_vision_projector(config)
        
        # Test forward pass
        batch_size, num_patches = 2, 196
        input_features = torch.randn(batch_size, num_patches, 768)
        
        output_features = projector(input_features)
        
        self.assertEqual(output_features.shape, (batch_size, num_patches, 2048))

    def test_mlp3x_projector(self):
        """Test 3-layer MLP projector."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="mlp3x_gelu"
        )
        
        projector = build_vision_projector(config)
        
        self.assertIsInstance(projector, nn.Sequential)
        # Should have 5 modules: Linear -> GELU -> Linear -> GELU -> Linear
        self.assertEqual(len(projector), 5)
        self.assertIsInstance(projector[0], nn.Linear)
        self.assertIsInstance(projector[1], nn.GELU)
        self.assertIsInstance(projector[2], nn.Linear)
        self.assertIsInstance(projector[3], nn.GELU)
        self.assertIsInstance(projector[4], nn.Linear)

    def test_resblock_projector(self):
        """Test ResBlock projector creation."""
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="resblock2"
        )
        
        projector = build_vision_projector(config)
        
        self.assertIsInstance(projector, nn.Sequential)
        # Should have 3 modules: Linear -> ResBlock -> ResBlock
        self.assertEqual(len(projector), 3)
        self.assertIsInstance(projector[0], nn.Linear)
        self.assertIsInstance(projector[1], SimpleResBlock)
        self.assertIsInstance(projector[2], SimpleResBlock)


if __name__ == "__main__":
    unittest.main(verbosity=2)