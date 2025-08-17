import os
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Any, Union
import logging

try:
    from transformers import Trainer, TrainingArguments
    from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


def maybe_zero_3(param, ignore_status=False, name=None):
    """Handle DeepSpeed ZeRO-3 parameter gathering."""
    try:
        from deepspeed import zero
        from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
        if hasattr(param, "ds_id"):
            if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
                if not ignore_status:
                    logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
            with zero.GatheredParameters([param]):
                param = param.data.detach().cpu().clone()
        else:
            param = param.detach().cpu().clone()
    except ImportError:
        # DeepSpeed not available, just clone the parameter
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    """Get multimodal adapter state dict, handling ZeRO-3 if available."""
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class MultimodalTrainer:
    """
    Custom trainer for multimodal models with two-stage training support.
    
    This trainer supports:
    1. Stage 1: Vision-language alignment (projector-only training)
    2. Stage 2: Instruction following (projector + language model training)
    """
    
    def __init__(self, 
                 model,
                 args: Optional[Dict] = None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 data_collator=None,
                 compute_metrics=None,
                 **kwargs):
        
        self.model = model
        self.args = args or {}
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        
        # Stage configuration
        self.stage = self.args.get('stage', 'stage1')
        self.tune_mm_mlp_adapter = self.args.get('tune_mm_mlp_adapter', False)
        self.freeze_mm_mlp_adapter = self.args.get('freeze_mm_mlp_adapter', False)
        
        # Initialize training components
        self._setup_training_components()
        
        # If transformers is available, use it as base
        if TRANSFORMERS_AVAILABLE:
            training_args = self._create_training_args()
            self.hf_trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
                **kwargs
            )
            # Override some methods
            self.hf_trainer.create_optimizer = self.create_optimizer
            self.hf_trainer.save_model = self.save_model
        else:
            self.hf_trainer = None
            logging.warning("Transformers not available, using basic training loop")
    
    def _create_training_args(self):
        """Create TrainingArguments from our config."""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        return TrainingArguments(
            output_dir=self.args.get('output_dir', './checkpoints'),
            num_train_epochs=self.args.get('num_train_epochs', 1),
            per_device_train_batch_size=self.args.get('per_device_train_batch_size', 4),
            per_device_eval_batch_size=self.args.get('per_device_eval_batch_size', 4),
            warmup_steps=self.args.get('warmup_steps', 500),
            weight_decay=self.args.get('weight_decay', 0.01),
            learning_rate=self.args.get('learning_rate', 5e-5),
            logging_dir=self.args.get('logging_dir', './logs'),
            logging_steps=self.args.get('logging_steps', 10),
            save_steps=self.args.get('save_steps', 500),
            eval_steps=self.args.get('eval_steps', 500),
            evaluation_strategy=self.args.get('evaluation_strategy', 'steps'),
            save_strategy=self.args.get('save_strategy', 'steps'),
            load_best_model_at_end=True,
            remove_unused_columns=False,  # Important for multimodal data
        )
    
    def _setup_training_components(self):
        """Set up training based on stage and configuration."""
        if self.stage == 'stage1' or self.tune_mm_mlp_adapter:
            self._setup_stage1_training()
        elif self.stage == 'stage2':
            self._setup_stage2_training()
        else:
            logging.warning(f"Unknown stage: {self.stage}, using default setup")
    
    def _setup_stage1_training(self):
        """
        Stage 1: Vision-language alignment.
        Freeze everything except the multimodal projector.
        """
        logging.info("Setting up Stage 1 training (vision-language alignment)")
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze multimodal projector
        if hasattr(self.model, 'mm_projector') and self.model.mm_projector is not None:
            for param in self.model.mm_projector.parameters():
                param.requires_grad = True
            logging.info("Unfroze multimodal projector parameters")
        elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'mm_projector'):
            # Handle wrapped models
            for param in self.model.get_model().mm_projector.parameters():
                param.requires_grad = True
            logging.info("Unfroze multimodal projector parameters (wrapped model)")
        else:
            logging.warning("No multimodal projector found in model")
        
        # Log trainable parameters
        self._log_trainable_parameters()
    
    def _setup_stage2_training(self):
        """
        Stage 2: Instruction following.
        Unfreeze projector and language model, keep vision encoder frozen.
        """
        logging.info("Setting up Stage 2 training (instruction following)")
        
        # Freeze vision tower
        if hasattr(self.model, 'vision_tower') and self.model.vision_tower is not None:
            for param in self.model.vision_tower.parameters():
                param.requires_grad = False
            logging.info("Froze vision tower parameters")
        elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'vision_tower'):
            # Handle wrapped models
            vision_tower = self.model.get_model().vision_tower
            if vision_tower is not None:
                for param in vision_tower.parameters():
                    param.requires_grad = False
                logging.info("Froze vision tower parameters (wrapped model)")
        
        # Unfreeze multimodal projector (if frozen)
        if self.freeze_mm_mlp_adapter:
            if hasattr(self.model, 'mm_projector') and self.model.mm_projector is not None:
                for param in self.model.mm_projector.parameters():
                    param.requires_grad = False
            elif hasattr(self.model, 'get_model') and hasattr(self.model.get_model(), 'mm_projector'):
                for param in self.model.get_model().mm_projector.parameters():
                    param.requires_grad = False
        
        # Log trainable parameters
        self._log_trainable_parameters()
    
    def _log_trainable_parameters(self):
        """Log information about trainable parameters."""
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        logging.info(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}%"
        )
    
    def create_optimizer(self):
        """
        Create optimizer with different learning rates for different components.
        """
        if not TRANSFORMERS_AVAILABLE:
            return self._create_basic_optimizer()
        
        # Use transformers' optimizer creation as base
        decay_parameters = get_parameter_names(self.model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        
        # Different learning rates for different components
        mm_projector_lr = self.args.get('mm_projector_lr', None)
        base_lr = self.args.get('learning_rate', 5e-5)
        
        optimizer_grouped_parameters = []
        
        # Multimodal projector parameters (higher learning rate)
        if mm_projector_lr is not None:
            mm_projector_params = []
            mm_projector_decay_params = []
            
            for name, param in self.model.named_parameters():
                if param.requires_grad and 'mm_projector' in name:
                    if name in decay_parameters:
                        mm_projector_decay_params.append(param)
                    else:
                        mm_projector_params.append(param)
            
            if mm_projector_params:
                optimizer_grouped_parameters.append({
                    "params": mm_projector_params,
                    "weight_decay": 0.0,
                    "lr": mm_projector_lr,
                })
            
            if mm_projector_decay_params:
                optimizer_grouped_parameters.append({
                    "params": mm_projector_decay_params,
                    "weight_decay": self.args.get('weight_decay', 0.01),
                    "lr": mm_projector_lr,
                })
        
        # Regular parameters (base learning rate)
        regular_params = []
        regular_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'mm_projector' not in name:
                if name in decay_parameters:
                    regular_decay_params.append(param)
                else:
                    regular_params.append(param)
        
        if regular_params:
            optimizer_grouped_parameters.append({
                "params": regular_params,
                "weight_decay": 0.0,
                "lr": base_lr,
            })
        
        if regular_decay_params:
            optimizer_grouped_parameters.append({
                "params": regular_decay_params,
                "weight_decay": self.args.get('weight_decay', 0.01),
                "lr": base_lr,
            })
        
        # Create optimizer
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(optimizer_grouped_parameters)
        
        return optimizer
    
    def _create_basic_optimizer(self):
        """Create basic optimizer when transformers is not available."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return torch.optim.AdamW(
            trainable_params, 
            lr=self.args.get('learning_rate', 5e-5),
            weight_decay=self.args.get('weight_decay', 0.01)
        )
    
    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """
        Save model with special handling for multimodal components.
        
        For stage 1, only save the multimodal projector.
        For stage 2, save the full model.
        """
        if output_dir is None:
            output_dir = self.args.get('output_dir', './checkpoints')
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.stage == 'stage1' or self.tune_mm_mlp_adapter:
            # Save only multimodal projector for stage 1
            self._save_mm_projector_only(output_dir)
        else:
            # Save full model for stage 2
            self._save_full_model(output_dir)
    
    def _save_mm_projector_only(self, output_dir: str):
        """Save only the multimodal projector (for stage 1)."""
        model = self.model
        if hasattr(model, 'get_model'):
            model = model.get_model()
        
        # Get multimodal projector state dict
        mm_projector_state = get_mm_adapter_state_maybe_zero_3(
            model.named_parameters(), 
            ['mm_projector']
        )
        
        if mm_projector_state:
            mm_projector_path = os.path.join(output_dir, "mm_projector.bin")
            torch.save(mm_projector_state, mm_projector_path)
            logging.info(f"Saved multimodal projector to {mm_projector_path}")
        else:
            logging.warning("No multimodal projector state found to save")
    
    def _save_full_model(self, output_dir: str):
        """Save the full model (for stage 2)."""
        if self.hf_trainer is not None:
            # Use transformers' save method
            self.hf_trainer.save_model(output_dir)
        else:
            # Basic save
            model_path = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(self.model.state_dict(), model_path)
            logging.info(f"Saved full model to {model_path}")
    
    def load_mm_projector_weights(self, checkpoint_path: str):
        """
        Load multimodal projector weights from checkpoint.
        
        Args:
            checkpoint_path: Path to the projector checkpoint file or directory
        """
        if os.path.isdir(checkpoint_path):
            # Look for mm_projector.bin in the directory
            projector_path = os.path.join(checkpoint_path, "mm_projector.bin")
        else:
            projector_path = checkpoint_path
            
        if not os.path.exists(projector_path):
            raise FileNotFoundError(f"Multimodal projector checkpoint not found at {projector_path}")
            
        logging.info(f"Loading multimodal projector weights from {projector_path}")
        
        # Load the state dict
        mm_projector_state = torch.load(projector_path, map_location='cpu')
        
        # Get the model (handle wrapped models)
        model = self.model
        if hasattr(model, 'get_model'):
            model = model.get_model()
            
        # Load the weights into the model
        missing_keys = []
        unexpected_keys = []
        
        for key, value in mm_projector_state.items():
            if hasattr(model, key.split('.')[0]):
                try:
                    # Navigate to the correct module
                    module = model
                    for attr in key.split('.')[:-1]:
                        module = getattr(module, attr)
                    param_name = key.split('.')[-1]
                    
                    if hasattr(module, param_name):
                        param = getattr(module, param_name)
                        if isinstance(param, torch.nn.Parameter):
                            param.data.copy_(value)
                        else:
                            setattr(module, param_name, value)
                    else:
                        missing_keys.append(key)
                except Exception as e:
                    logging.warning(f"Failed to load parameter {key}: {e}")
                    missing_keys.append(key)
            else:
                unexpected_keys.append(key)
        
        if missing_keys:
            logging.warning(f"Missing keys when loading projector weights: {missing_keys}")
        if unexpected_keys:
            logging.warning(f"Unexpected keys when loading projector weights: {unexpected_keys}")
            
        logging.info("Successfully loaded multimodal projector weights")
    
    def save_checkpoint(self, output_dir: str, step: Optional[int] = None):
        """
        Save a training checkpoint with optional step number.
        
        Args:
            output_dir: Base output directory
            step: Optional step number for checkpoint naming
        """
        if step is not None:
            checkpoint_dir = os.path.join(output_dir, f"checkpoint-{step}")
        else:
            checkpoint_dir = output_dir
            
        self.save_model(checkpoint_dir)
        
        # Save training state
        training_state = {
            'step': step,
            'stage': self.stage,
            'args': self.args,
        }
        
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        import json
        with open(state_path, 'w') as f:
            json.dump(training_state, f, indent=2)
            
        logging.info(f"Saved checkpoint to {checkpoint_dir}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """
        Load a training checkpoint.
        
        Args:
            checkpoint_dir: Directory containing the checkpoint
        """
        # Load training state
        state_path = os.path.join(checkpoint_dir, "training_state.json")
        if os.path.exists(state_path):
            import json
            with open(state_path, 'r') as f:
                training_state = json.load(f)
            logging.info(f"Loaded training state from step {training_state.get('step', 'unknown')}")
        
        # Load model weights
        if self.stage == 'stage1' or self.args.get('tune_mm_mlp_adapter', False):
            # Load only projector weights
            self.load_mm_projector_weights(checkpoint_dir)
        else:
            # Load full model
            model_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                logging.info(f"Loaded full model from {model_path}")
            else:
                logging.warning(f"No model file found at {model_path}")
    
    def find_latest_checkpoint(self, output_dir: str) -> Optional[str]:
        """
        Find the latest checkpoint in the output directory.
        
        Args:
            output_dir: Directory to search for checkpoints
            
        Returns:
            Path to the latest checkpoint directory, or None if not found
        """
        if not os.path.exists(output_dir):
            return None
            
        checkpoints = []
        for item in os.listdir(output_dir):
            if item.startswith('checkpoint-') and os.path.isdir(os.path.join(output_dir, item)):
                try:
                    step = int(item.split('-')[1])
                    checkpoints.append((step, os.path.join(output_dir, item)))
                except (IndexError, ValueError):
                    continue
        
        if checkpoints:
            # Return the checkpoint with the highest step number
            latest_checkpoint = max(checkpoints, key=lambda x: x[0])[1]
            logging.info(f"Found latest checkpoint: {latest_checkpoint}")
            return latest_checkpoint
        
        return None
    
    def train(self):
        """Start training."""
        if self.hf_trainer is not None:
            return self.hf_trainer.train()
        else:
            return self._basic_training_loop()
    
    def _basic_training_loop(self):
        """Basic training loop when transformers is not available."""
        logging.info("Starting basic training loop")
        
        # This is a simplified training loop for demonstration
        # In practice, you'd want a more sophisticated implementation
        optimizer = self._create_basic_optimizer()
        
        if self.train_dataset is None:
            logging.error("No training dataset provided")
            return
        
        from torch.utils.data import DataLoader
        
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.get('per_device_train_batch_size', 4),
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        self.model.train()
        
        num_epochs = self.args.get('num_train_epochs', 1)
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(dataloader):
                # Move batch to device
                device = next(self.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs
                
                # Backward pass
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                if step % self.args.get('logging_steps', 10) == 0:
                    logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
                
                if step % self.args.get('save_steps', 500) == 0:
                    output_dir = self.args.get('output_dir', './checkpoints')
                    global_step = epoch * len(dataloader) + step
                    self.save_checkpoint(output_dir, global_step)
        
        logging.info("Training completed")


class TrainingArguments:
    """Simple training arguments class for when transformers is not available."""
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def create_multimodal_trainer(
    model,
    stage: str = 'stage1',
    train_dataset=None,
    eval_dataset=None,
    tokenizer=None,
    data_collator=None,
    resume_from_checkpoint: Optional[str] = None,
    **training_args
) -> MultimodalTrainer:
    """
    Convenience function to create a multimodal trainer.
    
    Args:
        model: The multimodal model to train
        stage: Training stage ('stage1' or 'stage2')
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        tokenizer: Tokenizer
        data_collator: Data collator
        resume_from_checkpoint: Path to checkpoint to resume from
        **training_args: Additional training arguments
        
    Returns:
        Configured MultimodalTrainer instance
    """
    args = {'stage': stage}
    args.update(training_args)
    
    trainer = MultimodalTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Resume from checkpoint if specified
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'auto':
            # Find latest checkpoint automatically
            output_dir = args.get('output_dir', './checkpoints')
            latest_checkpoint = trainer.find_latest_checkpoint(output_dir)
            if latest_checkpoint:
                trainer.load_checkpoint(latest_checkpoint)
                logging.info(f"Automatically resumed from {latest_checkpoint}")
            else:
                logging.info("No checkpoint found for auto-resume")
        else:
            # Load specific checkpoint
            trainer.load_checkpoint(resume_from_checkpoint)
            logging.info(f"Resumed from checkpoint: {resume_from_checkpoint}")
    
    return trainer


def load_mm_projector_for_inference(model, checkpoint_path: str):
    """
    Load multimodal projector weights for inference.
    
    Args:
        model: The model to load weights into
        checkpoint_path: Path to the projector checkpoint
    """
    # Create a temporary trainer just for loading
    temp_trainer = MultimodalTrainer(model=model, args={'stage': 'stage1'})
    temp_trainer.load_mm_projector_weights(checkpoint_path)
    logging.info("Loaded multimodal projector for inference")