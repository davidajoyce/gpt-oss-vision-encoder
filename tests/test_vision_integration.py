import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
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


def test_transformer_initialization():
    """Test that Transformer initializes with vision components as None."""
    config = ModelConfig()
    model = Transformer(config, device=torch.device('cpu'))
    
    assert model.vision_tower is None
    assert model.mm_projector is None
    assert hasattr(model, 'config')


def test_forward_text_only():
    """Test text-only forward pass (backward compatibility)."""
    config = ModelConfig(
        num_hidden_layers=2,  # Small for testing
        hidden_size=512,
        vocab_size=1000
    )
    model = Transformer(config, device=torch.device('cpu'))
    
    input_ids = torch.randint(0, 1000, (2, 10))  # batch_size=2, seq_len=10
    output = model(input_ids)
    
    assert output.shape == (2, 10, 1000)  # [batch, seq, vocab]


@patch('gpt_oss.torch.vision_tower.build_vision_tower')
@patch('gpt_oss.torch.vision_projector.build_vision_projector')
def test_initialize_vision_modules(mock_projector_builder, mock_tower_builder):
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
    assert model.vision_tower is not None
    assert model.mm_projector is not None
    assert model.config.use_mm_proj is True
    assert model.config.mm_vision_tower == 'openai/clip-vit-large-patch14'
    assert model.config.mm_hidden_size == 1024


def test_get_vision_tower():
    """Test get_vision_tower method."""
    config = ModelConfig()
    model = Transformer(config, device=torch.device('cpu'))
    
    # Test when vision_tower is None
    assert model.get_vision_tower() is None
    
    # Test when vision_tower is a single object
    mock_tower = MockVisionTower()
    model.vision_tower = mock_tower
    assert model.get_vision_tower() is mock_tower
    
    # Test when vision_tower is a list (FSDP case)
    model.vision_tower = [mock_tower]
    assert model.get_vision_tower() is mock_tower


def test_forward_multimodal_basic():
    """Test basic multimodal forward pass."""
    config = ModelConfig(
        num_hidden_layers=1,  # Minimal for testing
        hidden_size=256,
        vocab_size=1000
    )
    model = Transformer(config, device=torch.device('cpu'))
    
    # Set up mock vision components
    model.vision_tower = MockVisionTower(hidden_size=256)
    model.mm_projector = nn.Identity()  # Simple passthrough
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (1, 5))  # batch_size=1, seq_len=5
    images = torch.randn(1, 3, 224, 224)  # batch_size=1, channels=3, height=224, width=224
    
    output = model(input_ids, images=images)
    
    # Output should have combined sequence length (image patches + text tokens)
    # 196 image patches + 5 text tokens = 201 total sequence length
    assert output.shape[0] == 1  # batch size
    assert output.shape[1] == 196 + 5  # seq len (image patches + text)
    assert output.shape[2] == 1000  # vocab size


def test_forward_without_images():
    """Test that forward without images uses text-only path."""
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=256,
        vocab_size=1000
    )
    model = Transformer(config, device=torch.device('cpu'))
    
    # Set up vision components but don't use them
    model.vision_tower = MockVisionTower()
    model.mm_projector = nn.Linear(1024, 256)
    
    input_ids = torch.randint(0, 1000, (1, 5))
    output = model(input_ids)  # No images provided
    
    # Should be text-only output
    assert output.shape == (1, 5, 1000)


def test_forward_without_vision_tower():
    """Test that forward with images but no vision tower uses text-only path."""
    config = ModelConfig(
        num_hidden_layers=1,
        hidden_size=256,
        vocab_size=1000
    )
    model = Transformer(config, device=torch.device('cpu'))
    
    input_ids = torch.randint(0, 1000, (1, 5))
    images = torch.randn(1, 3, 224, 224)
    output = model(input_ids, images=images)  # Images provided but no vision tower
    
    # Should fall back to text-only output
    assert output.shape == (1, 5, 1000)


if __name__ == "__main__":
    pytest.main([__file__])