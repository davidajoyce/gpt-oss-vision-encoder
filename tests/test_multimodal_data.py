#!/usr/bin/env python3
"""
Test cases for multimodal data pipeline.
"""

import unittest
import tempfile
import shutil
import os
import json
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import numpy as np

# Import our modules
import sys
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.train.multimodal_data import (
    DataArguments,
    LazySupervisedDataset,
    DataCollatorForSupervisedDataset,
    make_supervised_data_module,
    preprocess_multimodal,
    tokenizer_image_token,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
    IGNORE_INDEX
)


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.model_max_length = 512
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
    def __call__(self, text, **kwargs):
        if isinstance(text, list):
            # Handle batch of texts
            input_ids = []
            for t in text:
                # Simple tokenization: split by spaces and assign IDs
                tokens = t.split()
                ids = [hash(token) % 1000 + 10 for token in tokens]  # Avoid special token IDs
                input_ids.append(torch.tensor(ids))
            
            if kwargs.get('return_tensors') == 'pt':
                max_len = max(len(ids) for ids in input_ids)
                padded = torch.zeros(len(input_ids), max_len, dtype=torch.long)
                for i, ids in enumerate(input_ids):
                    padded[i, :len(ids)] = ids
                return type('TokenizerOutput', (), {'input_ids': padded})()
            else:
                return type('TokenizerOutput', (), {'input_ids': input_ids})()
        else:
            # Handle single text
            tokens = text.split()
            ids = [hash(token) % 1000 + 10 for token in tokens]
            if kwargs.get('return_tensors') == 'pt':
                return type('TokenizerOutput', (), {'input_ids': torch.tensor(ids)})()
            else:
                return type('TokenizerOutput', (), {'input_ids': ids})()


class TestMultimodalData(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.tokenizer = MockTokenizer()
        
        # Create test data
        self.test_data = [
            {
                "id": "1",
                "image": "test1.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_IMAGE_TOKEN}\nWhat do you see in this image?"
                    },
                    {
                        "from": "gpt", 
                        "value": "I see a beautiful landscape with mountains."
                    }
                ]
            },
            {
                "id": "2",
                "conversations": [
                    {
                        "from": "human",
                        "value": "What is the capital of France?"
                    },
                    {
                        "from": "gpt",
                        "value": "The capital of France is Paris."
                    }
                ]
            }
        ]
        
        # Save test data to file
        self.data_path = os.path.join(self.temp_dir, 'test_data.json')
        with open(self.data_path, 'w') as f:
            json.dump(self.test_data, f)
        
        # Create test images
        self.image_folder = os.path.join(self.temp_dir, 'images')
        os.makedirs(self.image_folder)
        
        # Create a test image
        img = Image.new('RGB', (224, 224), color='red')
        img.save(os.path.join(self.image_folder, 'test1.jpg'))
        
        # Data arguments
        self.data_args = DataArguments(
            data_path=self.data_path,
            is_multimodal=True,
            image_folder=self.image_folder,
            lazy_preprocess=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_data_arguments_creation(self):
        """Test DataArguments creation and defaults."""
        # Test with minimal arguments
        args = DataArguments()
        self.assertIsNone(args.data_path)
        self.assertFalse(args.lazy_preprocess)
        self.assertFalse(args.is_multimodal)
        self.assertEqual(args.image_aspect_ratio, 'square')
        
        # Test with custom arguments
        args = DataArguments(
            data_path='/path/to/data',
            is_multimodal=True,
            image_aspect_ratio='pad'
        )
        self.assertEqual(args.data_path, '/path/to/data')
        self.assertTrue(args.is_multimodal)
        self.assertEqual(args.image_aspect_ratio, 'pad')
    
    def test_preprocess_multimodal(self):
        """Test multimodal preprocessing function."""
        sources = [
            [
                {
                    "from": "human",
                    "value": f"{DEFAULT_IMAGE_TOKEN}\nDescribe this image"
                },
                {
                    "from": "gpt",
                    "value": "This is a test image."
                }
            ]
        ]
        
        # Test with multimodal enabled
        data_args = DataArguments(is_multimodal=True)
        result = preprocess_multimodal(sources, data_args)
        
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        
        # Check that image token was processed
        human_message = result[0][0]["value"]
        self.assertIn(DEFAULT_IMAGE_TOKEN, human_message)
    
    def test_tokenizer_image_token(self):
        """Test the image token tokenization function."""
        # Test prompt with image token
        prompt = f"Please describe {DEFAULT_IMAGE_TOKEN} in detail."
        
        result = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX)
        
        self.assertIsInstance(result, list)
        self.assertIn(IMAGE_TOKEN_INDEX, result)
        
        # Test with return_tensors
        result_tensor = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        )
        self.assertIsInstance(result_tensor, torch.Tensor)
        self.assertTrue((result_tensor == IMAGE_TOKEN_INDEX).any())
    
    @patch('gpt_oss.train.multimodal_data.rank0_print')
    def test_lazy_supervised_dataset_creation(self, mock_print):
        """Test LazySupervisedDataset creation."""
        dataset = LazySupervisedDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.tokenizer, self.tokenizer)
        self.assertEqual(dataset.data_args, self.data_args)
        
        # Test lengths property
        lengths = dataset.lengths
        self.assertEqual(len(lengths), 2)
        self.assertIsInstance(lengths[0], int)
        
        # Test modality_lengths property
        modality_lengths = dataset.modality_lengths
        self.assertEqual(len(modality_lengths), 2)
        # First sample has image (positive), second doesn't (negative)
        self.assertGreater(modality_lengths[0], 0)  # Has image
        self.assertLess(modality_lengths[1], 0)     # No image
    
    @patch('gpt_oss.train.multimodal_data.rank0_print')
    def test_lazy_supervised_dataset_getitem(self, mock_print):
        """Test LazySupervisedDataset __getitem__ method."""
        dataset = LazySupervisedDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        
        # Test getting item with image
        item0 = dataset[0]
        self.assertIn('input_ids', item0)
        self.assertIn('labels', item0)
        self.assertIn('image', item0)
        
        self.assertIsInstance(item0['input_ids'], torch.Tensor)
        self.assertIsInstance(item0['labels'], torch.Tensor)
        self.assertIsInstance(item0['image'], torch.Tensor)
        
        # Test getting item without image
        item1 = dataset[1]
        self.assertIn('input_ids', item1)
        self.assertIn('labels', item1)
        self.assertNotIn('image', item1)  # No image for this item
    
    @patch('gpt_oss.train.multimodal_data.rank0_print')
    def test_lazy_supervised_dataset_missing_image(self, mock_print):
        """Test dataset behavior with missing image file."""
        # Add data with non-existent image
        bad_data = [
            {
                "id": "bad",
                "image": "nonexistent.jpg",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{DEFAULT_IMAGE_TOKEN}\nWhat's in this image?"
                    },
                    {
                        "from": "gpt",
                        "value": "I cannot see the image."
                    }
                ]
            }
        ]
        
        bad_data_path = os.path.join(self.temp_dir, 'bad_data.json')
        with open(bad_data_path, 'w') as f:
            json.dump(bad_data, f)
        
        bad_data_args = DataArguments(
            data_path=bad_data_path,
            is_multimodal=True,
            image_folder=self.image_folder,
        )
        
        dataset = LazySupervisedDataset(
            data_path=bad_data_path,
            tokenizer=self.tokenizer,
            data_args=bad_data_args
        )
        
        # Should handle missing image gracefully
        item = dataset[0]
        self.assertIn('image', item)
        # Should be a zero tensor (placeholder)
        self.assertTrue(torch.all(item['image'] == 0))
    
    def test_data_collator(self):
        """Test DataCollatorForSupervisedDataset."""
        collator = DataCollatorForSupervisedDataset(self.tokenizer)
        
        # Create test instances
        instances = [
            {
                'input_ids': torch.tensor([10, 20, 30]),
                'labels': torch.tensor([10, 20, 30]),
                'image': torch.rand(3, 224, 224)
            },
            {
                'input_ids': torch.tensor([40, 50]),
                'labels': torch.tensor([40, 50]),
                'image': torch.rand(3, 224, 224)
            },
            {
                'input_ids': torch.tensor([60, 70, 80, 90]),
                'labels': torch.tensor([60, 70, 80, 90]),
                # No image for this instance
            }
        ]
        
        batch = collator(instances)
        
        # Check output structure
        self.assertIn('input_ids', batch)
        self.assertIn('labels', batch)
        self.assertIn('attention_mask', batch)
        self.assertIn('images', batch)
        
        # Check shapes
        batch_size = len(instances)
        max_seq_len = 4  # Longest sequence
        
        self.assertEqual(batch['input_ids'].shape, (batch_size, max_seq_len))
        self.assertEqual(batch['labels'].shape, (batch_size, max_seq_len))
        self.assertEqual(batch['attention_mask'].shape, (batch_size, max_seq_len))
        self.assertEqual(batch['images'].shape, (batch_size, 3, 224, 224))
        
        # Check padding
        self.assertEqual(batch['input_ids'][0, 3].item(), self.tokenizer.pad_token_id)  # Padded
        self.assertEqual(batch['labels'][1, 2].item(), IGNORE_INDEX)  # Padded with ignore index
        
        # Check attention mask
        self.assertEqual(batch['attention_mask'][0, 2].item(), 1)  # Real token
        self.assertEqual(batch['attention_mask'][0, 3].item(), 0)  # Padded token
    
    def test_data_collator_no_images(self):
        """Test data collator with no images."""
        collator = DataCollatorForSupervisedDataset(self.tokenizer)
        
        instances = [
            {
                'input_ids': torch.tensor([10, 20]),
                'labels': torch.tensor([10, 20]),
            },
            {
                'input_ids': torch.tensor([30, 40, 50]),
                'labels': torch.tensor([30, 40, 50]),
            }
        ]
        
        batch = collator(instances)
        
        # Should not have images key when no images present
        self.assertNotIn('images', batch)
        
        # Should still have other keys
        self.assertIn('input_ids', batch)
        self.assertIn('labels', batch)
        self.assertIn('attention_mask', batch)
    
    @patch('gpt_oss.train.multimodal_data.rank0_print')
    def test_make_supervised_data_module(self, mock_print):
        """Test make_supervised_data_module function."""
        result = make_supervised_data_module(
            tokenizer=self.tokenizer,
            data_args=self.data_args
        )
        
        self.assertIn('train_dataset', result)
        self.assertIn('eval_dataset', result)
        self.assertIn('data_collator', result)
        
        self.assertIsInstance(result['train_dataset'], LazySupervisedDataset)
        self.assertIsNone(result['eval_dataset'])  # Default None
        self.assertIsInstance(result['data_collator'], DataCollatorForSupervisedDataset)


class TestConversationHandling(unittest.TestCase):
    """Test conversation handling utilities."""
    
    def test_simple_conversation(self):
        """Test SimpleConversation class."""
        from gpt_oss.train.multimodal_data import SimpleConversation
        
        conv = SimpleConversation()
        
        # Test initial state
        self.assertEqual(conv.roles, ["USER", "ASSISTANT"])
        self.assertEqual(conv.messages, [])
        
        # Test adding messages
        conv.append_message("USER", "Hello, how are you?")
        conv.append_message("ASSISTANT", "I'm doing well, thank you!")
        
        self.assertEqual(len(conv.messages), 2)
        self.assertEqual(conv.messages[0], ["USER", "Hello, how are you?"])
        
        # Test getting prompt
        prompt = conv.get_prompt()
        self.assertIn("USER: Hello, how are you?", prompt)
        self.assertIn("ASSISTANT: I'm doing well, thank you!", prompt)
        
        # Test copy
        conv_copy = conv.copy()
        self.assertEqual(conv_copy.messages, conv.messages)
        self.assertIsNot(conv_copy.messages, conv.messages)  # Different objects


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)