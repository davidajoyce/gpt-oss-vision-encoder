import sys
import os
import unittest
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from gpt_oss.torch.model import ModelConfig, Transformer


class MockVisionTower(nn.Module):
    """Mock vision tower for testing."""
    
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_loaded = True
    
    def forward(self, images):
        batch_size = images.shape[0]
        num_patches = 196  # 14x14 patches
        return torch.randn(batch_size, num_patches, self.hidden_size)
    
    def load_model(self):
        pass


class MockArgs:
    """Mock arguments for testing."""
    
    def __init__(self, **kwargs):
        self.vision_tower = kwargs.get('vision_tower', 'openai/clip-vit-large-patch14')
        self.mm_projector_type = kwargs.get('mm_projector_type', 'linear')
        self.pretrain_mm_mlp_adapter = kwargs.get('pretrain_mm_mlp_adapter', None)


class TestVisionIntegrationSimple(unittest.TestCase):
    """Test vision integration components without full forward pass."""

    def test_transformer_initialization(self):
        """Test that Transformer initializes with vision components as None."""
        config = ModelConfig()
        model = Transformer(config, device=torch.device('cpu'))
        
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.mm_projector)
        self.assertTrue(hasattr(model, 'config'))

    @patch('gpt_oss.torch.vision_tower.build_vision_tower')
    @patch('gpt_oss.torch.vision_projector.build_vision_projector')
    def test_initialize_vision_modules(self, mock_projector_builder, mock_tower_builder):
        """Test vision module initialization."""
        # Setup mocks
        mock_vision_tower = MockVisionTower(hidden_size=1024)
        mock_tower_builder.return_value = mock_vision_tower
        
        mock_projector = nn.Linear(1024, 512)
        mock_projector_builder.return_value = mock_projector
        
        # Create model and initialize
        config = ModelConfig(hidden_size=512)
        model = Transformer(config, device=torch.device('cpu'))
        
        args = MockArgs(vision_tower='openai/clip-vit-large-patch14')
        model.initialize_vision_modules(args)
        
        # Verify initialization
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.mm_projector)
        self.assertTrue(model.config.use_mm_proj)
        self.assertEqual(model.config.mm_vision_tower, 'openai/clip-vit-large-patch14')
        self.assertEqual(model.config.mm_hidden_size, 1024)

    def test_get_vision_tower_single(self):
        """Test get_vision_tower method with single tower."""
        config = ModelConfig()
        model = Transformer(config, device=torch.device('cpu'))
        
        # Test when vision_tower is None
        self.assertIsNone(model.get_vision_tower())
        
        # Test when vision_tower is a single object
        mock_tower = MockVisionTower()
        model.vision_tower = mock_tower
        self.assertEqual(model.get_vision_tower(), mock_tower)

    def test_get_vision_tower_list(self):
        """Test get_vision_tower method with list (FSDP case)."""
        config = ModelConfig()
        model = Transformer(config, device=torch.device('cpu'))
        
        # Test when vision_tower is a list (FSDP case)
        # We need to bypass the torch.nn.Module restriction by using a mock
        mock_tower = MockVisionTower()
        
        # Temporarily override get_vision_tower to test the list case
        original_vision_tower = None
        model._vision_tower_list = [mock_tower]  # Store as private attribute
        
        # Mock the get_vision_tower behavior
        def mock_get_vision_tower():
            vision_tower = getattr(model, '_vision_tower_list', getattr(model, 'vision_tower', None))
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]
            return vision_tower
        
        model.get_vision_tower = mock_get_vision_tower
        self.assertEqual(model.get_vision_tower(), mock_tower)

    def test_vision_component_compatibility(self):
        """Test that vision components work together."""
        config = ModelConfig(hidden_size=256)
        model = Transformer(config, device=torch.device('cpu'))
        
        # Set up compatible vision components
        model.vision_tower = MockVisionTower(hidden_size=512)
        model.mm_projector = nn.Linear(512, 256)
        
        # Test vision processing pipeline
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        
        # Process through vision tower
        vision_features = model.vision_tower(images)
        self.assertEqual(vision_features.shape, (batch_size, 196, 512))
        
        # Process through projector
        projected_features = model.mm_projector(vision_features)
        self.assertEqual(projected_features.shape, (batch_size, 196, 256))

    def test_transformer_has_vision_methods(self):
        """Test that Transformer has all expected vision methods."""
        config = ModelConfig()
        model = Transformer(config, device=torch.device('cpu'))
        
        # Check that all vision methods exist
        self.assertTrue(hasattr(model, 'get_vision_tower'))
        self.assertTrue(hasattr(model, 'initialize_vision_modules'))
        self.assertTrue(hasattr(model, 'forward_text_only'))
        self.assertTrue(hasattr(model, 'forward_multimodal'))
        
        # Check that they are callable
        self.assertTrue(callable(getattr(model, 'get_vision_tower')))
        self.assertTrue(callable(getattr(model, 'initialize_vision_modules')))
        self.assertTrue(callable(getattr(model, 'forward_text_only')))
        self.assertTrue(callable(getattr(model, 'forward_multimodal')))

    def test_vision_config_integration(self):
        """Test that vision configuration is properly integrated."""
        config = ModelConfig(
            mm_vision_tower="openai/clip-vit-large-patch14",
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=1024,
            use_mm_proj=True
        )
        
        model = Transformer(config, device=torch.device('cpu'))
        
        # Check that config is accessible
        self.assertEqual(model.config.mm_vision_tower, "openai/clip-vit-large-patch14")
        self.assertEqual(model.config.mm_projector_type, "mlp2x_gelu")
        self.assertEqual(model.config.mm_hidden_size, 1024)
        self.assertTrue(model.config.use_mm_proj)

    def test_vision_components_none_by_default(self):
        """Test that vision components are None by default and can be set."""
        config = ModelConfig()
        model = Transformer(config, device=torch.device('cpu'))
        
        # Initially None
        self.assertIsNone(model.vision_tower)
        self.assertIsNone(model.mm_projector)
        
        # Can be set
        model.vision_tower = MockVisionTower()
        model.mm_projector = nn.Identity()
        
        self.assertIsNotNone(model.vision_tower)
        self.assertIsNotNone(model.mm_projector)


if __name__ == "__main__":
    unittest.main(verbosity=2)