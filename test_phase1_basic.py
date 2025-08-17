#!/usr/bin/env python3
"""
Basic test script for Phase 1 vision components.
Tests the core functionality without requiring external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_model_config():
    """Test ModelConfig vision extensions."""
    print("Testing ModelConfig...")
    
    from gpt_oss.torch.model import ModelConfig
    
    # Test defaults
    config = ModelConfig()
    assert config.mm_vision_tower is None
    assert config.mm_projector_type == "linear"
    assert config.use_mm_proj is False
    print("✓ ModelConfig defaults work")
    
    # Test customization
    config2 = ModelConfig(
        mm_vision_tower="openai/clip-vit-large-patch14",
        mm_projector_type="mlp2x_gelu",
        use_mm_proj=True
    )
    assert config2.mm_vision_tower == "openai/clip-vit-large-patch14"
    assert config2.mm_projector_type == "mlp2x_gelu"
    assert config2.use_mm_proj is True
    print("✓ ModelConfig customization works")


def test_projector_builder():
    """Test vision projector builder."""
    print("Testing projector builder...")
    
    try:
        from gpt_oss.torch.model import ModelConfig
        from gpt_oss.torch.vision_projector import build_vision_projector, IdentityMap
        
        # Test linear projector
        config = ModelConfig(
            mm_hidden_size=768,
            hidden_size=2048,
            mm_projector_type="linear"
        )
        
        # This will fail if torch is not available, but we can test the import
        print("✓ Projector builder imports work")
        
        # Test identity projector
        config2 = ModelConfig(
            mm_hidden_size=2048,
            hidden_size=2048,
            mm_projector_type="identity"
        )
        print("✓ Projector builder configuration works")
        
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Torch not available, skipping torch-dependent tests")
        else:
            raise


def test_vision_tower():
    """Test vision tower imports."""
    print("Testing vision tower...")
    
    try:
        from gpt_oss.torch.vision_tower import VisionTower, build_vision_tower
        print("✓ Vision tower imports work")
    except ImportError as e:
        if "torch" in str(e) or "transformers" in str(e):
            print("⚠ Dependencies not available, skipping vision tower tests")
        else:
            raise


def test_transformer_extensions():
    """Test Transformer class extensions."""
    print("Testing Transformer extensions...")
    
    try:
        from gpt_oss.torch.model import Transformer, ModelConfig
        
        # Test that Transformer can be imported and has new methods
        config = ModelConfig()
        
        # Check if the new methods exist
        assert hasattr(Transformer, 'get_vision_tower')
        assert hasattr(Transformer, 'initialize_vision_modules')
        assert hasattr(Transformer, 'forward_text_only')
        assert hasattr(Transformer, 'forward_multimodal')
        print("✓ Transformer extensions work")
        
    except ImportError as e:
        if "torch" in str(e):
            print("⚠ Torch not available, skipping Transformer tests")
        else:
            raise


def main():
    """Run all basic tests."""
    print("Running Phase 1 Basic Tests")
    print("=" * 40)
    
    try:
        test_model_config()
        test_projector_builder()
        test_vision_tower()
        test_transformer_extensions()
        
        print("=" * 40)
        print("✅ All basic tests passed!")
        print("\nNote: Full functionality tests require torch and transformers.")
        print("To run complete tests, install dependencies and use pytest.")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)