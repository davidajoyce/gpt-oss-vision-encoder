#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import torch
from transformers import AutoTokenizer
from gpt_oss.torch.model import Transformer, ModelConfig
from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, add_image_tokens, validate_image_tokens

def test_image_token_constants():
    """Test that image token constants are properly defined"""
    print("=== Testing Image Token Constants ===")
    
    from gpt_oss.constants import (
        IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
        DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    )
    
    # Check constants are defined
    assert IMAGE_TOKEN_INDEX == -200, f"Expected -200, got {IMAGE_TOKEN_INDEX}"
    assert DEFAULT_IMAGE_TOKEN == "<image>", f"Expected '<image>', got {DEFAULT_IMAGE_TOKEN}"
    assert IGNORE_INDEX == -100, f"Expected -100, got {IGNORE_INDEX}"
    
    print("✅ All image token constants properly defined")

def test_tokenizer_image_token_addition():
    """Test adding image tokens to tokenizer"""
    print("\n=== Testing Image Token Addition ===")
    
    # Create a simple tokenizer for testing
    try:
        # Try to load a real tokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    except:
        print("Warning: Could not load real tokenizer, creating mock")
        # Create mock tokenizer if real one not available
        class MockTokenizer:
            def __init__(self):
                self.vocab = {"<pad>": 0, "<unk>": 1, "hello": 2, "world": 3}
                self.unk_token_id = 1
                
            def __len__(self):
                return len(self.vocab)
                
            def add_special_tokens(self, tokens_dict):
                added = 0
                for token in tokens_dict["additional_special_tokens"]:
                    if token not in self.vocab:
                        self.vocab[token] = len(self.vocab)
                        added += 1
                return added
                
            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, self.unk_token_id)
                
            def decode(self, token_ids):
                reverse_vocab = {v: k for k, v in self.vocab.items()}
                return " ".join([reverse_vocab.get(tid, "<unk>") for tid in token_ids])
        
        tokenizer = MockTokenizer()
    
    original_size = len(tokenizer)
    print(f"Original vocabulary size: {original_size}")
    
    # Add image tokens
    image_token_id = add_image_tokens(tokenizer)
    
    new_size = len(tokenizer)
    print(f"New vocabulary size: {new_size}")
    
    # Validate addition
    assert new_size > original_size, "Vocabulary size should have increased"
    assert image_token_id is not None, "Image token ID should be returned"
    
    # Validate the token can be retrieved
    retrieved_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    assert retrieved_id == image_token_id, f"Token ID mismatch: {retrieved_id} vs {image_token_id}"
    
    print("✅ Image token addition successful")
    return tokenizer

def test_model_tokenizer_integration():
    """Test model integration with image tokenizer"""
    print("\n=== Testing Model-Tokenizer Integration ===")
    
    # Create a small test config
    config = ModelConfig(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64
    )
    
    # Create model
    model = Transformer(config, device='cpu')
    
    # Create tokenizer  
    tokenizer = test_tokenizer_image_token_addition()
    
    # Test model initialization with image tokens
    original_vocab_size = model.config.vocab_size
    print(f"Model original vocab size: {original_vocab_size}")
    
    # Initialize image tokenizer
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    
    # Check that model vocabulary was updated
    new_vocab_size = model.config.vocab_size
    print(f"Model new vocab size: {new_vocab_size}")
    
    assert new_vocab_size >= original_vocab_size, "Model vocab size should not decrease"
    assert image_token_id is not None, "Image token ID should be returned"
    
    # Test that embeddings were resized properly
    assert model.embedding.num_embeddings == new_vocab_size, "Embedding layer not properly resized"
    assert model.unembedding.out_features == new_vocab_size, "Output layer not properly resized"
    
    print("✅ Model-tokenizer integration successful")
    return model, tokenizer, image_token_id

def test_token_embedding_functionality():
    """Test that image tokens can be embedded and processed"""
    print("\n=== Testing Token Embedding Functionality ===")
    
    model, tokenizer, image_token_id = test_model_tokenizer_integration()
    
    # Create test input with image token
    test_tokens = torch.tensor([[image_token_id, 1, 2, 3]], dtype=torch.long)
    print(f"Test tokens: {test_tokens}")
    
    # Test embedding
    embeddings = model.embed_tokens(test_tokens)
    print(f"Embeddings shape: {embeddings.shape}")
    
    expected_shape = (1, 4, model.config.hidden_size)
    assert embeddings.shape == expected_shape, f"Expected shape {expected_shape}, got {embeddings.shape}"
    
    # Check that image token embedding is not zero (should be initialized)
    image_token_embedding = embeddings[0, 0, :]  # First token in sequence
    assert not torch.allclose(image_token_embedding, torch.zeros_like(image_token_embedding)), \
        "Image token embedding should not be zero"
    
    print("✅ Token embedding functionality working")

def test_validation_functions():
    """Test image token validation functions"""
    print("\n=== Testing Validation Functions ===")
    
    model, tokenizer, image_token_id = test_model_tokenizer_integration()
    
    # Test validation
    is_valid = validate_image_tokens(tokenizer)
    assert is_valid, "Image tokens should validate successfully"
    
    print("✅ Validation functions working")

def run_all_tests():
    """Run all image token tests"""
    print("🚀 Starting Image Token Infrastructure Tests")
    print("=" * 50)
    
    try:
        test_image_token_constants()
        test_tokenizer_image_token_addition()  
        test_model_tokenizer_integration()
        test_token_embedding_functionality()
        test_validation_functions()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED! Image token infrastructure is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)