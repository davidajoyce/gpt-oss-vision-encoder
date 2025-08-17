#!/usr/bin/env python3
"""
Multimodal text generation example for GPT-OSS.

This script demonstrates how to generate text responses from both text prompts and images
using the LLaVA-style multimodal architecture.

Example usage:
    # Text-only generation (backward compatible)
    python -m gpt_oss.generate_multimodal -p "What is the capital of France?" model/
    
    # Multimodal generation
    python -m gpt_oss.generate_multimodal -p "What do you see in this image?" \
        --image path/to/image.jpg model/
    
    # With vision tower and projector
    python -m gpt_oss.generate_multimodal -p "Describe the scene" \
        --image image.jpg --vision-tower openai/clip-vit-base-patch32 \
        --projector-path projector_weights.bin model/
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, List, Iterator, Tuple

import torch
from PIL import Image
import numpy as np

from gpt_oss.tokenizer import get_tokenizer


class MultimodalTokenGenerator:
    """Enhanced TokenGenerator that supports both text-only and multimodal inference."""
    
    @torch.inference_mode()
    def __init__(self, 
                 checkpoint: str, 
                 device: torch.device,
                 vision_tower: Optional[str] = None,
                 projector_path: Optional[str] = None):
        """
        Initialize multimodal token generator.
        
        Args:
            checkpoint: Path to model checkpoint
            device: Device to run on
            vision_tower: Vision tower model name (e.g., 'openai/clip-vit-base-patch32')
            projector_path: Path to projector weights (.bin file)
        """
        self.device = device
        
        # Load base model
        from gpt_oss.torch.model import Transformer
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)
        
        # Initialize vision components if specified
        if vision_tower is not None:
            self._initialize_vision_components(vision_tower, projector_path)
        else:
            logging.info("No vision tower specified - text-only mode")
        
        self.multimodal_enabled = hasattr(self.model, 'vision_tower') and self.model.vision_tower is not None
        
    def _initialize_vision_components(self, vision_tower: str, projector_path: Optional[str]):
        """Initialize vision tower and projector."""
        try:
            # Initialize vision modules
            from gpt_oss.torch.vision_tower import build_vision_tower
            from gpt_oss.torch.vision_projector import build_vision_projector
            from gpt_oss.torch.model import ModelConfig
            
            # Update model config for vision
            if not hasattr(self.model.config, 'mm_vision_tower'):
                self.model.config.mm_vision_tower = vision_tower
                self.model.config.use_mm_proj = True
                self.model.config.mm_projector_type = 'linear'
                
            # Build vision tower
            self.model.vision_tower = build_vision_tower(self.model.config)
            
            # Build projector
            self.model.mm_projector = build_vision_projector(self.model.config)
            
            # Load projector weights if provided
            if projector_path is not None:
                self._load_projector_weights(projector_path)
                
            logging.info(f"Initialized vision components: {vision_tower}")
            
        except Exception as e:
            logging.error(f"Failed to initialize vision components: {e}")
            raise
    
    def _load_projector_weights(self, projector_path: str):
        """Load projector weights from checkpoint."""
        try:
            projector_state = torch.load(projector_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'mm_projector.weight' in projector_state:
                # Direct projector state dict
                self.model.mm_projector.load_state_dict(projector_state)
            else:
                # Nested state dict
                projector_keys = {k: v for k, v in projector_state.items() if 'mm_projector' in k}
                if projector_keys:
                    # Remove 'mm_projector.' prefix
                    clean_state = {k.replace('mm_projector.', ''): v for k, v in projector_keys.items()}
                    self.model.mm_projector.load_state_dict(clean_state)
                else:
                    logging.warning(f"No projector weights found in {projector_path}")
            
            logging.info(f"Loaded projector weights from {projector_path}")
            
        except Exception as e:
            logging.error(f"Failed to load projector weights: {e}")
            # Continue without pre-trained projector weights
    
    def process_image(self, image_input) -> torch.Tensor:
        """
        Process an image into tensor format using the robust image processor.
        
        Args:
            image_input: Image in various formats (path, PIL Image, numpy array, etc.)
            
        Returns:
            Processed image tensor ready for vision tower
        """
        try:
            # Use robust image processor
            from gpt_oss.vision.image_processor import create_processor_for_vision_tower
            
            # Get vision tower name for processor configuration
            vision_tower_name = getattr(self.model.config, 'mm_vision_tower', 'clip')
            processor = create_processor_for_vision_tower(vision_tower_name)
            
            # Process image and move to device
            tensor = processor.process(image_input, return_tensors='pt')
            return tensor.to(self.device)
                
        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            raise
    
    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: List[int],
                 stop_tokens: List[int],
                 image: Optional[str] = None,
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False) -> Iterator[Tuple[int, Optional[float]]]:
        """
        Generate tokens from prompt and optional image.
        
        Args:
            prompt_tokens: Tokenized text prompt
            stop_tokens: Tokens that signal end of generation
            image: Optional image (path, PIL Image, numpy array, etc.)
            temperature: Sampling temperature (0.0 = greedy)
            max_tokens: Maximum tokens to generate (0 = unlimited)
            return_logprobs: Whether to return log probabilities
            
        Yields:
            (token_id, log_probability) tuples
        """
        # Process image if provided
        image_tensor = None
        if image is not None:
            if not self.multimodal_enabled:
                logging.warning("Image provided but multimodal mode not enabled - ignoring image")
            else:
                image_tensor = self.process_image(image)
        
        # Initialize token sequence
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        
        # Generate tokens one by one
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # Forward pass through model
            input_tensor = torch.as_tensor(tokens, dtype=torch.int32, device=self.device)
            
            if image_tensor is not None and self.multimodal_enabled:
                # Multimodal forward pass
                logits = self.model(input_tensor, images=image_tensor)[-1]
            else:
                # Text-only forward pass (backward compatible)
                logits = self.model(input_tensor)[-1]
            
            # Sample next token
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            
            tokens.append(predicted_token)
            num_generated_tokens += 1
            
            # Calculate log probability if requested
            logprob = None
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                logprob = logprobs[predicted_token].item()
            
            yield predicted_token, logprob
            
            # Check for stop tokens
            if predicted_token in stop_tokens:
                break


def main(args):
    """Main generation function."""
    # Set up logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize device and backend
    match args.backend:
        case "torch":
            from gpt_oss.torch.utils import init_distributed
            device = init_distributed()
            
            # Create multimodal generator
            generator = MultimodalTokenGenerator(
                checkpoint=args.checkpoint,
                device=device,
                vision_tower=args.vision_tower,
                projector_path=args.projector_path
            )
            
        case "triton":
            # TODO: Implement multimodal Triton backend
            raise NotImplementedError("Multimodal Triton backend not yet implemented")
            
        case "vllm":
            # TODO: Implement multimodal vLLM backend  
            raise NotImplementedError("Multimodal vLLM backend not yet implemented")
            
        case _:
            raise ValueError(f"Invalid backend: {args.backend}")
    
    # Get tokenizer
    tokenizer = get_tokenizer()
    
    # Encode prompt
    tokens = tokenizer.encode(args.prompt)
    max_tokens = None if args.limit == 0 else args.limit
    
    # Validate image path if provided
    if args.image and not Path(args.image).exists():
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # Generate response
    print(f"Prompt: {repr(args.prompt)}")
    if args.image:
        print(f"Image: {args.image}")
    print("Response: ", end="", flush=True)
    
    generated_tokens = []
    for token, logprob in generator.generate(
        prompt_tokens=tokens,
        stop_tokens=[tokenizer.eot_token],
        image=args.image,
        temperature=args.temperature,
        max_tokens=max_tokens,
        return_logprobs=args.verbose
    ):
        generated_tokens.append(token)
        token_text = tokenizer.decode([token])
        
        if args.verbose:
            print(f"\nGenerated token: {repr(token_text)}, logprob: {logprob:.4f}")
        else:
            print(token_text, end="", flush=True)
    
    print()  # Final newline
    
    if args.verbose:
        print(f"\nGenerated {len(generated_tokens)} tokens")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multimodal text generation example",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        "checkpoint",
        metavar="CHECKPOINT",
        type=str,
        help="Path to the model checkpoint"
    )
    
    # Text generation arguments
    parser.add_argument(
        "-p", "--prompt",
        metavar="PROMPT", 
        type=str,
        default="Hello, how are you?",
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "-t", "--temperature",
        metavar="TEMP",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    
    parser.add_argument(
        "-l", "--limit",
        metavar="LIMIT",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (0 = unlimited)"
    )
    
    # Multimodal arguments
    parser.add_argument(
        "--image",
        metavar="IMAGE_PATH",
        type=str,
        help="Path to image file for multimodal generation"
    )
    
    parser.add_argument(
        "--vision-tower",
        metavar="MODEL_NAME",
        type=str,
        help="Vision tower model name (e.g., 'openai/clip-vit-base-patch32')"
    )
    
    parser.add_argument(
        "--projector-path",
        metavar="PATH",
        type=str,
        help="Path to trained projector weights (.bin file)"
    )
    
    # Backend arguments
    parser.add_argument(
        "-b", "--backend",
        metavar="BACKEND",
        type=str,
        default="torch",
        choices=["torch", "triton", "vllm"],
        help="Inference backend (only 'torch' supports multimodal currently)"
    )
    
    # Other arguments
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with token-by-token details"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image and not args.vision_tower:
        parser.error("--vision-tower is required when --image is specified")
    
    if args.backend != "torch" and (args.image or args.vision_tower):
        parser.error(f"Multimodal inference only supported with torch backend, got {args.backend}")
    
    main(args)