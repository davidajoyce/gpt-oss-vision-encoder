import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from gpt_oss.torch.model import ModelConfig


class TestVisionConfig:
    """Test vision-related configuration extensions to ModelConfig."""
    
    def test_model_config_vision_defaults(self):
        """Test that ModelConfig has proper vision defaults."""
        config = ModelConfig()
        
        # Check vision-related defaults
        assert config.mm_vision_tower is None
        assert config.mm_projector_type == "linear"
        assert config.mm_hidden_size is None
        assert config.mm_vision_select_layer == -2
        assert config.mm_vision_select_feature == "patch"
        assert config.mm_patch_merge_type == "flat"
        assert config.use_mm_proj is False
        assert config.tune_mm_mlp_adapter is False
        assert config.freeze_mm_mlp_adapter is False
        assert config.pretrain_mm_mlp_adapter is None

    def test_model_config_vision_customization(self):
        """Test that vision parameters can be customized."""
        config = ModelConfig(
            mm_vision_tower="openai/clip-vit-large-patch14",
            mm_projector_type="mlp2x_gelu",
            mm_hidden_size=1024,
            use_mm_proj=True
        )
        
        assert config.mm_vision_tower == "openai/clip-vit-large-patch14"
        assert config.mm_projector_type == "mlp2x_gelu"
        assert config.mm_hidden_size == 1024
        assert config.use_mm_proj is True

    def test_model_config_from_dict(self):
        """Test that ModelConfig can be created from dict (JSON compatibility)."""
        config_dict = {
            "num_hidden_layers": 24,
            "hidden_size": 2048,
            "mm_vision_tower": "openai/clip-vit-base-patch32",
            "mm_projector_type": "linear",
            "mm_hidden_size": 768,
            "use_mm_proj": True
        }
        
        config = ModelConfig(**config_dict)
        
        assert config.num_hidden_layers == 24
        assert config.hidden_size == 2048
        assert config.mm_vision_tower == "openai/clip-vit-base-patch32"
        assert config.mm_projector_type == "linear"
        assert config.mm_hidden_size == 768
        assert config.use_mm_proj is True

    def test_model_config_backward_compatibility(self):
        """Test that ModelConfig maintains backward compatibility."""
        # Test that we can still create configs without vision parameters
        config = ModelConfig(
            num_hidden_layers=12,
            hidden_size=768,
            vocab_size=50000
        )
        
        assert config.num_hidden_layers == 12
        assert config.hidden_size == 768
        assert config.vocab_size == 50000
        # Vision defaults should still be present
        assert config.mm_vision_tower is None
        assert config.use_mm_proj is False


if __name__ == "__main__":
    import unittest
    
    # Convert pytest-style class to unittest
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVisionConfig)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if result.wasSuccessful():
        print("\n✅ All vision config tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)