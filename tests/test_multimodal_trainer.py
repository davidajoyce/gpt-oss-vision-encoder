#!/usr/bin/env python3
"""
Test cases for multimodal trainer functionality.
"""

import unittest
import tempfile
import shutil
import os
import json
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

# Import our modules
import sys
sys.path.insert(0, '/Users/davidj/dev/gpt-oss-vision-encoder')

from gpt_oss.torch.model import ModelConfig, Transformer
from gpt_oss.train.multimodal_trainer import MultimodalTrainer, create_multimodal_trainer
from gpt_oss.train.multimodal_data import DataArguments


class MockDataset:
    """Mock dataset for testing."""
    
    def __init__(self, size=10):
        self.size = size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 1000, (20,)),
            'labels': torch.randint(0, 1000, (20,)),
            'attention_mask': torch.ones(20),
        }


class MockDataCollator:
    """Mock data collator for testing."""
    
    def __call__(self, instances):
        batch_size = len(instances)
        max_len = max(len(inst['input_ids']) for inst in instances)
        
        input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
        
        for i, inst in enumerate(instances):
            length = len(inst['input_ids'])
            input_ids[i, :length] = inst['input_ids']
            labels[i, :length] = inst['labels']
            attention_mask[i, :length] = inst['attention_mask']
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }


class TestMultimodalTrainer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
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
        
        # Create model with multimodal components
        self.model = Transformer(self.config)
        
        # Add mm_projector for testing
        self.model.mm_projector = nn.Linear(512, 128)
        
        # Mock datasets
        self.train_dataset = MockDataset(20)
        self.eval_dataset = MockDataset(5)
        self.data_collator = MockDataCollator()
        
        # Training args
        self.training_args = {
            'output_dir': self.temp_dir,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 2,
            'learning_rate': 1e-4,
            'save_steps': 5,
            'logging_steps': 1,
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization_stage1(self):
        """Test trainer initialization for stage 1."""
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage1', **self.training_args},
            train_dataset=self.train_dataset,
            data_collator=self.data_collator
        )
        
        self.assertEqual(trainer.stage, 'stage1')
        
        # Check that only mm_projector parameters are trainable
        trainable_params = [name for name, param in self.model.named_parameters() if param.requires_grad]
        self.assertTrue(any('mm_projector' in name for name in trainable_params))
        
        # Check that other parameters are frozen
        for name, param in self.model.named_parameters():
            if 'mm_projector' not in name:
                self.assertFalse(param.requires_grad, f"Parameter {name} should be frozen in stage 1")
    
    def test_trainer_initialization_stage2(self):
        """Test trainer initialization for stage 2."""
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage2', **self.training_args},
            train_dataset=self.train_dataset,
            data_collator=self.data_collator
        )
        
        self.assertEqual(trainer.stage, 'stage2')
        
        # In stage 2, language model and projector should be trainable
        # (vision tower should be frozen, but we don't have one in this test)
        trainable_count = sum(1 for param in self.model.parameters() if param.requires_grad)
        self.assertGreater(trainable_count, 0)
    
    def test_save_and_load_mm_projector(self):
        """Test saving and loading multimodal projector weights."""
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage1', **self.training_args},
        )
        
        # Save original projector weights
        original_weight = self.model.mm_projector.weight.data.clone()
        original_bias = self.model.mm_projector.bias.data.clone()
        
        # Save projector
        save_dir = os.path.join(self.temp_dir, 'projector_test')
        os.makedirs(save_dir, exist_ok=True)  # Create directory first
        trainer._save_mm_projector_only(save_dir)
        
        # Check that the file was created
        projector_path = os.path.join(save_dir, 'mm_projector.bin')
        self.assertTrue(os.path.exists(projector_path))
        
        # Modify the weights
        self.model.mm_projector.weight.data.fill_(0.5)
        self.model.mm_projector.bias.data.fill_(0.1)
        
        # Load the weights back
        trainer.load_mm_projector_weights(save_dir)
        
        # Check that weights were restored
        torch.testing.assert_close(self.model.mm_projector.weight.data, original_weight)
        torch.testing.assert_close(self.model.mm_projector.bias.data, original_bias)
    
    def test_checkpoint_saving_and_loading(self):
        """Test checkpoint saving and loading."""
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage1', **self.training_args},
        )
        
        # Save a checkpoint
        step = 100
        trainer.save_checkpoint(self.temp_dir, step)
        
        # Check that checkpoint directory was created
        checkpoint_dir = os.path.join(self.temp_dir, f'checkpoint-{step}')
        self.assertTrue(os.path.isdir(checkpoint_dir))
        
        # Check that training state was saved
        state_file = os.path.join(checkpoint_dir, 'training_state.json')
        self.assertTrue(os.path.exists(state_file))
        
        with open(state_file, 'r') as f:
            state = json.load(f)
        self.assertEqual(state['step'], step)
        self.assertEqual(state['stage'], 'stage1')
        
        # Test loading checkpoint
        trainer.load_checkpoint(checkpoint_dir)
    
    def test_find_latest_checkpoint(self):
        """Test finding the latest checkpoint."""
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage1', **self.training_args},
        )
        
        # Create some checkpoints
        steps = [10, 50, 30]
        for step in steps:
            trainer.save_checkpoint(self.temp_dir, step)
        
        # Find latest checkpoint
        latest = trainer.find_latest_checkpoint(self.temp_dir)
        
        # Should find checkpoint-50 (highest step)
        expected = os.path.join(self.temp_dir, 'checkpoint-50')
        self.assertEqual(latest, expected)
        
        # Test with no checkpoints
        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir)
        latest_empty = trainer.find_latest_checkpoint(empty_dir)
        self.assertIsNone(latest_empty)
    
    def test_create_multimodal_trainer_convenience(self):
        """Test the convenience function for creating trainers."""
        trainer = create_multimodal_trainer(
            model=self.model,
            stage='stage1',
            train_dataset=self.train_dataset,
            data_collator=self.data_collator,
            **self.training_args
        )
        
        self.assertIsInstance(trainer, MultimodalTrainer)
        self.assertEqual(trainer.stage, 'stage1')
        self.assertEqual(trainer.train_dataset, self.train_dataset)
    
    def test_resume_from_checkpoint_auto(self):
        """Test auto-resuming from the latest checkpoint."""
        # First, create some checkpoints
        trainer1 = create_multimodal_trainer(
            model=self.model,
            stage='stage1',
            **self.training_args
        )
        trainer1.save_checkpoint(self.temp_dir, 25)
        trainer1.save_checkpoint(self.temp_dir, 50)
        
        # Now create a new trainer that should auto-resume
        trainer2 = create_multimodal_trainer(
            model=self.model,
            stage='stage1',
            resume_from_checkpoint='auto',
            **self.training_args
        )
        
        # The trainer should have loaded from the latest checkpoint
        self.assertIsNotNone(trainer2)
    
    def test_optimizer_creation_with_different_lr(self):
        """Test optimizer creation with different learning rates for components."""
        training_args_with_mm_lr = {
            **self.training_args,
            'mm_projector_lr': 1e-3,
            'learning_rate': 1e-5,
        }
        
        # Test the basic optimizer creation without transformers complexity
        trainer = MultimodalTrainer(
            model=self.model,
            args={'stage': 'stage2', **training_args_with_mm_lr},
        )
        
        optimizer = trainer.create_optimizer()
        
        # Should have at least one parameter group
        self.assertGreaterEqual(len(optimizer.param_groups), 1)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
    
    @patch('gpt_oss.train.multimodal_trainer.TRANSFORMERS_AVAILABLE', False)
    def test_basic_training_loop(self):
        """Test the basic training loop when transformers is not available."""
        training_args_short = {
            **self.training_args,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 1,
            'logging_steps': 1,
            'save_steps': 10,  # Set high to avoid saving during test
        }
        
        # Create a simple model that returns a loss
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 1)
                self.mm_projector = nn.Linear(10, 10)
            
            def forward(self, **kwargs):
                # Return a simple loss for testing
                return torch.tensor(1.0, requires_grad=True)
            
            def parameters(self):
                # Return an iterator, not a list
                for param in self.linear.parameters():
                    yield param
                for param in self.mm_projector.parameters():
                    yield param
            
            def named_parameters(self):
                for name, param in self.linear.named_parameters():
                    yield f'linear.{name}', param
                for name, param in self.mm_projector.named_parameters():
                    yield f'mm_projector.{name}', param
        
        simple_model = SimpleModel()
        
        # Create a very small dataset for quick testing
        small_dataset = MockDataset(2)
        
        trainer = MultimodalTrainer(
            model=simple_model,
            args={'stage': 'stage1', **training_args_short},
            train_dataset=small_dataset,
            data_collator=self.data_collator
        )
        
        # This should complete without errors
        trainer._basic_training_loop()


class TestMultimodalData(unittest.TestCase):
    """Test multimodal data handling."""
    
    def test_data_arguments(self):
        """Test DataArguments dataclass."""
        data_args = DataArguments(
            data_path='/path/to/data.json',
            is_multimodal=True,
            image_folder='/path/to/images',
        )
        
        self.assertEqual(data_args.data_path, '/path/to/data.json')
        self.assertTrue(data_args.is_multimodal)
        self.assertEqual(data_args.image_folder, '/path/to/images')


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)