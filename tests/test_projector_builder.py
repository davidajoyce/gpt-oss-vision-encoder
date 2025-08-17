import pytest
import torch
import torch.nn as nn
from gpt_oss.torch.model import ModelConfig
from gpt_oss.torch.vision_projector import build_vision_projector, IdentityMap, SimpleResBlock


def test_linear_projector():
    """Test linear projector creation."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="linear"
    )
    
    projector = build_vision_projector(config)
    
    assert isinstance(projector, nn.Linear)
    assert projector.in_features == 768
    assert projector.out_features == 2048


def test_mlp_projector():
    """Test MLP projector creation."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="mlp2x_gelu"
    )
    
    projector = build_vision_projector(config)
    
    assert isinstance(projector, nn.Sequential)
    # Should have 3 modules: Linear -> GELU -> Linear
    assert len(projector) == 3
    assert isinstance(projector[0], nn.Linear)
    assert isinstance(projector[1], nn.GELU)
    assert isinstance(projector[2], nn.Linear)
    
    # Check dimensions
    assert projector[0].in_features == 768
    assert projector[0].out_features == 2048
    assert projector[2].in_features == 2048
    assert projector[2].out_features == 2048


def test_identity_projector():
    """Test identity projector creation."""
    config = ModelConfig(
        mm_hidden_size=2048,
        hidden_size=2048,
        mm_projector_type="identity"
    )
    
    projector = build_vision_projector(config)
    
    assert isinstance(projector, IdentityMap)


def test_identity_projector_dimension_mismatch():
    """Test that identity projector raises error when dimensions don't match."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="identity"
    )
    
    with pytest.raises(ValueError, match="Identity projector requires"):
        build_vision_projector(config)


def test_projector_missing_mm_hidden_size():
    """Test that projector builder raises error when mm_hidden_size is missing."""
    config = ModelConfig(
        hidden_size=2048,
        mm_projector_type="linear"
    )
    
    with pytest.raises(ValueError, match="mm_hidden_size must be set"):
        build_vision_projector(config)


def test_unknown_projector_type():
    """Test that unknown projector type raises error."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="unknown_type"
    )
    
    with pytest.raises(ValueError, match="Unknown projector type"):
        build_vision_projector(config)


def test_projector_forward_pass():
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
    
    assert output_features.shape == (batch_size, num_patches, 2048)


def test_mlp3x_projector():
    """Test 3-layer MLP projector."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="mlp3x_gelu"
    )
    
    projector = build_vision_projector(config)
    
    assert isinstance(projector, nn.Sequential)
    # Should have 5 modules: Linear -> GELU -> Linear -> GELU -> Linear
    assert len(projector) == 5
    assert isinstance(projector[0], nn.Linear)
    assert isinstance(projector[1], nn.GELU)
    assert isinstance(projector[2], nn.Linear)
    assert isinstance(projector[3], nn.GELU)
    assert isinstance(projector[4], nn.Linear)


def test_resblock_projector():
    """Test ResBlock projector creation."""
    config = ModelConfig(
        mm_hidden_size=768,
        hidden_size=2048,
        mm_projector_type="resblock2"
    )
    
    projector = build_vision_projector(config)
    
    assert isinstance(projector, nn.Sequential)
    # Should have 3 modules: Linear -> ResBlock -> ResBlock
    assert len(projector) == 3
    assert isinstance(projector[0], nn.Linear)
    assert isinstance(projector[1], SimpleResBlock)
    assert isinstance(projector[2], SimpleResBlock)


if __name__ == "__main__":
    pytest.main([__file__])