#!/usr/bin/env python3

import sys
import os
sys.path.append('.')

import torch
from transformers import AutoTokenizer
from gpt_oss.torch.model import Transformer, ModelConfig
from gpt_oss.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, IGNORE_INDEX

def create_test_model_and_tokenizer():
    """Create a test model with vision components and tokenizer"""
    print("=== Setting up test model and tokenizer ===")
    
    # Create model config
    config = ModelConfig(
        num_hidden_layers=2,
        hidden_size=64,
        vocab_size=1000,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=64,
        # Vision config
        mm_vision_tower="openai/clip-vit-base-patch32",
        mm_projector_type="linear",
        use_mm_proj=True
    )
    
    # Create model
    model = Transformer(config, device='cpu')
    
    # Create tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    except:
        # Create mock tokenizer if real one not available
        class MockTokenizer:
            def __init__(self):
                self.vocab = {"<pad>": 0, "<unk>": 1, "What": 2, "is": 3, "this": 4, "?": 5}
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
            
            def encode(self, text, return_tensors=None):
                # Simple mock encoding
                tokens = text.split()
                token_ids = [self.convert_tokens_to_ids(token) for token in tokens]
                if return_tensors == "pt":
                    return torch.tensor([token_ids])
                return token_ids
        
        tokenizer = MockTokenizer()
    
    # Initialize image tokenizer
    image_token_id = model.initialize_image_tokenizer(tokenizer)
    
    # Make sure the model knows the correct image token ID
    model.set_image_token_id(image_token_id)
    
    # Initialize vision components (mock)
    try:
        model.initialize_vision_modules()
    except:
        print("Warning: Could not initialize real vision modules, using mock")
        # Create mock vision components
        class MockVisionTower:
            def __init__(self):
                self.hidden_size = 768
                
            def __call__(self, images):
                batch_size = images.shape[0]
                # Return mock features: [batch, num_patches, vision_dim]
                return torch.randn(batch_size, 49, 768)  # 7x7 patches
        
        class MockProjector:
            def __call__(self, features):
                # Project from 768 to model hidden size (64)
                batch_size, num_patches, _ = features.shape
                return torch.randn(batch_size, num_patches, 64)
        
        model.vision_tower = MockVisionTower()
        model.mm_projector = MockProjector()
    
    print("✅ Test model and tokenizer ready")
    return model, tokenizer

def test_text_only_backward_compatibility():
    """Test that text-only input still works"""
    print("\n=== Testing Text-Only Backward Compatibility ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    
    # Create text-only input
    text_input = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)  # "What is this ?"
    labels = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    
    # Test with no images
    result = model.prepare_multimodal_inputs(text_input, images=None, labels=labels)
    
    # Validate result structure
    assert "inputs_embeds" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Check shapes
    expected_shape = (1, 4, model.config.hidden_size)
    assert result["inputs_embeds"].shape == expected_shape
    assert result["attention_mask"].shape == (1, 4)
    assert result["labels"].shape == (1, 4)
    
    # Check that labels are preserved
    assert torch.equal(result["labels"], labels)
    
    print("✅ Text-only backward compatibility working")

def test_single_image_processing():
    """Test processing a sequence with a single image token"""
    print("\n=== Testing Single Image Processing ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    # Create input with image token: "What is <image> ?"
    input_ids = torch.tensor([[2, 3, image_token_id, 4, 5]], dtype=torch.long)
    labels = torch.tensor([[2, 3, IGNORE_INDEX, 4, 5]], dtype=torch.long)  # Ignore image token
    
    # Create mock image
    images = torch.randn(1, 3, 224, 224)  # Single image
    
    # Process
    result = model.prepare_multimodal_inputs(input_ids, images=images, labels=labels)
    
    # Validate result structure
    assert "inputs_embeds" in result
    assert "attention_mask" in result
    assert "labels" in result
    
    # Check that sequence length increased (image token replaced with multiple patches)
    original_seq_len = input_ids.shape[1]
    new_seq_len = result["inputs_embeds"].shape[1]
    print(f"Original sequence length: {original_seq_len}")
    print(f"New sequence length: {new_seq_len}")
    assert new_seq_len > original_seq_len, "Sequence should be longer after adding image features"
    
    # Check that image positions have IGNORE_INDEX in labels
    labels_result = result["labels"]
    ignore_positions = (labels_result == IGNORE_INDEX).sum().item()
    print(f"Number of IGNORE_INDEX positions: {ignore_positions}")
    assert ignore_positions > 1, "Should have multiple IGNORE_INDEX positions for image patches"
    
    print("✅ Single image processing working")

def test_multiple_images_processing():
    """Test processing a sequence with multiple image tokens"""
    print("\n=== Testing Multiple Images Processing ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    # Create input with two image tokens: "What <image> is <image> ?"
    input_ids = torch.tensor([[2, image_token_id, 3, image_token_id, 4]], dtype=torch.long)
    labels = torch.tensor([[2, IGNORE_INDEX, 3, IGNORE_INDEX, 4]], dtype=torch.long)
    
    # Create mock images (batch of 1, but sequence contains 2 image tokens)
    images = torch.randn(1, 3, 224, 224)  # We'll reuse this image for both positions
    
    # Process
    result = model.prepare_multimodal_inputs(input_ids, images=images, labels=labels)
    
    # Check that sequence length increased significantly
    original_seq_len = input_ids.shape[1]
    new_seq_len = result["inputs_embeds"].shape[1]
    print(f"Original sequence length: {original_seq_len}")
    print(f"New sequence length: {new_seq_len}")
    
    # Should have added features for 2 images
    expected_increase = 2 * 49 - 2  # 2 * num_patches - 2 original image tokens
    expected_seq_len = original_seq_len + expected_increase
    assert new_seq_len == expected_seq_len, f"Expected {expected_seq_len}, got {new_seq_len}"
    
    print("✅ Multiple images processing working")

def test_batch_processing():
    """Test processing a batch with mixed content"""
    print("\n=== Testing Batch Processing ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    # Create batch with different sequence types
    # Sequence 1: Text only: "What is this ?"
    # Sequence 2: With image: "What is <image> ?"
    batch_ids = [
        torch.tensor([2, 3, 4, 5], dtype=torch.long),  # text only
        torch.tensor([2, 3, image_token_id, 4], dtype=torch.long)  # with image
    ]
    
    batch_labels = [
        torch.tensor([2, 3, 4, 5], dtype=torch.long),
        torch.tensor([2, 3, IGNORE_INDEX, 4], dtype=torch.long)
    ]
    
    # Pad to same length for batch processing
    max_len = max(len(seq) for seq in batch_ids)
    padded_ids = torch.zeros(2, max_len, dtype=torch.long)
    padded_labels = torch.full((2, max_len), IGNORE_INDEX, dtype=torch.long)
    
    for i, (ids, labels) in enumerate(zip(batch_ids, batch_labels)):
        padded_ids[i, :len(ids)] = ids
        padded_labels[i, :len(labels)] = labels
    
    # Create images (batch size 2)
    images = torch.randn(2, 3, 224, 224)
    
    # Process
    result = model.prepare_multimodal_inputs(padded_ids, images=images, labels=padded_labels)
    
    # Validate batch processing
    assert result["inputs_embeds"].shape[0] == 2, "Should maintain batch size"
    assert result["attention_mask"].shape[0] == 2, "Should maintain batch size"
    assert result["labels"].shape[0] == 2, "Should maintain batch size"
    
    # Check that sequences have different lengths due to image processing
    seq1_len = result["attention_mask"][0].sum().item()
    seq2_len = result["attention_mask"][1].sum().item()
    print(f"Sequence 1 length: {seq1_len}")
    print(f"Sequence 2 length: {seq2_len}")
    
    # Sequence 2 should be longer (has image)
    assert seq2_len > seq1_len, "Sequence with image should be longer"
    
    print("✅ Batch processing working")

def test_edge_cases():
    """Test edge cases and error conditions"""
    print("\n=== Testing Edge Cases ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    # Test 1: Empty sequence
    try:
        empty_ids = torch.tensor([[]], dtype=torch.long)
        result = model.prepare_multimodal_inputs(empty_ids, images=None)
        print("✅ Empty sequence handled")
    except Exception as e:
        print(f"❌ Empty sequence failed: {e}")
    
    # Test 2: Image token at beginning
    start_image_ids = torch.tensor([[image_token_id, 2, 3]], dtype=torch.long)
    images = torch.randn(1, 3, 224, 224)
    result = model.prepare_multimodal_inputs(start_image_ids, images=images)
    assert result["inputs_embeds"].shape[1] > 3, "Should handle image token at start"
    print("✅ Image token at beginning handled")
    
    # Test 3: Image token at end
    end_image_ids = torch.tensor([[2, 3, image_token_id]], dtype=torch.long)
    result = model.prepare_multimodal_inputs(end_image_ids, images=images)
    assert result["inputs_embeds"].shape[1] > 3, "Should handle image token at end"
    print("✅ Image token at end handled")
    
    # Test 4: Only image token
    only_image_ids = torch.tensor([[image_token_id]], dtype=torch.long)
    result = model.prepare_multimodal_inputs(only_image_ids, images=images)
    assert result["inputs_embeds"].shape[1] > 1, "Should handle only image token"
    print("✅ Only image token handled")
    
    print("✅ Edge cases working")

def test_label_alignment():
    """Test that labels are properly aligned with embeddings"""
    print("\n=== Testing Label Alignment ===")
    
    model, tokenizer = create_test_model_and_tokenizer()
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    # Create input: "What <image> is this ?"
    input_ids = torch.tensor([[2, image_token_id, 3, 4, 5]], dtype=torch.long)
    labels = torch.tensor([[2, IGNORE_INDEX, 3, 4, 5]], dtype=torch.long)
    
    images = torch.randn(1, 3, 224, 224)
    
    # Process
    result = model.prepare_multimodal_inputs(input_ids, images=images, labels=labels)
    
    # Check that labels have same length as embeddings
    embeds_len = result["inputs_embeds"].shape[1]
    labels_len = result["labels"].shape[1]
    assert embeds_len == labels_len, f"Embeddings length {embeds_len} != labels length {labels_len}"
    
    # Check that image region has IGNORE_INDEX
    labels_result = result["labels"][0]
    attention_mask = result["attention_mask"][0]
    
    # Count non-padding positions
    valid_positions = attention_mask.sum().item()
    
    # Count IGNORE positions
    ignore_count = (labels_result[:valid_positions] == IGNORE_INDEX).sum().item()
    print(f"Valid positions: {valid_positions}")
    print(f"IGNORE_INDEX positions: {ignore_count}")
    
    # Should have more IGNORE positions than the original 1 (due to image patches)
    assert ignore_count > 1, "Should have multiple IGNORE positions for image patches"
    
    print("✅ Label alignment working")

def run_all_tests():
    """Run all multimodal input tests"""
    print("🚀 Starting Multimodal Input Preparation Tests")
    print("=" * 60)
    
    try:
        test_text_only_backward_compatibility()
        test_single_image_processing()
        test_multiple_images_processing()
        test_batch_processing()
        test_edge_cases()
        test_label_alignment()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Multimodal input preparation is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)