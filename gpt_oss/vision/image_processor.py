"""
Image processing pipeline for GPT-OSS multimodal inference.

This module provides robust image preprocessing that handles various image formats,
sizes, and quality levels for consistent input to vision models.
"""

import io
import logging
from pathlib import Path
from typing import Union, Optional, Tuple, List
import base64

import torch
import numpy as np
from PIL import Image, ImageOps, ExifTags
import torchvision.transforms as transforms


class ImageProcessor:
    """
    Robust image processor that handles various image formats and preprocessing needs.
    
    Supports:
    - Multiple input formats (file paths, PIL Images, numpy arrays, bytes, base64)
    - Automatic format detection and conversion
    - EXIF orientation correction
    - Flexible resizing strategies
    - Vision model-specific preprocessing
    """
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 resize_mode: str = 'resize',
                 normalize: bool = True,
                 mean: List[float] = None,
                 std: List[float] = None):
        """
        Initialize image processor.
        
        Args:
            target_size: Target (height, width) for processed images
            resize_mode: How to handle resizing ('resize', 'crop', 'pad')
            normalize: Whether to apply normalization
            mean: Normalization mean values (defaults to ImageNet)
            std: Normalization std values (defaults to ImageNet)
        """
        self.target_size = target_size
        self.resize_mode = resize_mode
        self.normalize = normalize
        
        # Default to ImageNet normalization values
        self.mean = mean or [0.485, 0.456, 0.406]
        self.std = std or [0.229, 0.224, 0.225]
        
        # Build transform pipeline
        self._build_transforms()
        
        # Supported image formats
        self.supported_formats = {
            '.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'
        }
        
        logging.info(f"Initialized ImageProcessor: {target_size}, mode={resize_mode}")
    
    def _build_transforms(self):
        """Build the torchvision transform pipeline."""
        transform_list = []
        
        # Resize transformation
        if self.resize_mode == 'resize':
            transform_list.append(transforms.Resize(self.target_size))
        elif self.resize_mode == 'crop':
            transform_list.append(transforms.Resize(min(self.target_size)))
            transform_list.append(transforms.CenterCrop(self.target_size))
        elif self.resize_mode == 'pad':
            transform_list.append(transforms.Resize(min(self.target_size)))
            transform_list.append(transforms.Pad(
                padding=self._calculate_padding(),
                fill=0,
                padding_mode='constant'
            ))
        
        # Convert to tensor
        transform_list.append(transforms.ToTensor())
        
        # Normalization
        if self.normalize:
            transform_list.append(transforms.Normalize(mean=self.mean, std=self.std))
        
        self.transform = transforms.Compose(transform_list)
    
    def _calculate_padding(self) -> Tuple[int, int, int, int]:
        """Calculate padding for pad mode (not used in basic resize)."""
        # This would be implemented for more sophisticated padding
        return (0, 0, 0, 0)
    
    def _correct_orientation(self, image: Image.Image) -> Image.Image:
        """
        Correct image orientation based on EXIF data.
        
        Args:
            image: PIL Image object
            
        Returns:
            Orientation-corrected PIL Image
        """
        try:
            # Get EXIF orientation tag
            for orientation in ExifTags.TAGS.keys():
                if ExifTags.TAGS[orientation] == 'Orientation':
                    break
            
            if hasattr(image, '_getexif'):
                exif = image._getexif()
                if exif is not None and orientation in exif:
                    orientation_value = exif[orientation]
                    
                    # Apply rotation based on orientation value
                    if orientation_value == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation_value == 6:
                        image = image.rotate(270, expand=True)
                    elif orientation_value == 8:
                        image = image.rotate(90, expand=True)
        
        except (AttributeError, KeyError, TypeError):
            # If EXIF processing fails, just return the original image
            pass
        
        return image
    
    def _load_from_path(self, image_path: Union[str, Path]) -> Image.Image:
        """Load image from file path."""
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        
        if path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported image format: {path.suffix}")
        
        try:
            image = Image.open(path)
            image = self._correct_orientation(image)
            return image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image {path}: {e}")
    
    def _load_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """Load image from raw bytes."""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image = self._correct_orientation(image)
            return image.convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to load image from bytes: {e}")
    
    def _load_from_base64(self, base64_string: str) -> Image.Image:
        """Load image from base64 encoded string."""
        try:
            # Handle data URLs
            if base64_string.startswith('data:image'):
                base64_string = base64_string.split(',')[1]
            
            image_bytes = base64.b64decode(base64_string)
            return self._load_from_bytes(image_bytes)
        except Exception as e:
            raise ValueError(f"Failed to load image from base64: {e}")
    
    def _load_from_array(self, image_array: np.ndarray) -> Image.Image:
        """Load image from numpy array."""
        try:
            # Handle different array shapes and types
            if image_array.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            # Handle grayscale (H, W) -> (H, W, 1)
            if len(image_array.shape) == 2:
                image_array = np.expand_dims(image_array, axis=2)
            
            # Handle different channel orders
            if image_array.shape[2] == 1:
                # Grayscale -> RGB
                image_array = np.repeat(image_array, 3, axis=2)
            elif image_array.shape[2] == 4:
                # RGBA -> RGB
                image_array = image_array[:, :, :3]
            
            return Image.fromarray(image_array)
        except Exception as e:
            raise ValueError(f"Failed to load image from array: {e}")
    
    def load_image(self, 
                   image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
                   validate: bool = True) -> Image.Image:
        """
        Load and standardize image from various input types.
        
        Args:
            image_input: Image in various formats
            validate: Whether to validate image properties
            
        Returns:
            Standardized PIL Image in RGB format
        """
        try:
            # Handle different input types
            if isinstance(image_input, (str, Path)):
                if isinstance(image_input, str) and image_input.startswith(('data:', 'iVBOR', '/9j/', 'R0lGOD')):
                    # Base64 encoded image
                    image = self._load_from_base64(image_input)
                else:
                    # File path
                    image = self._load_from_path(image_input)
            
            elif isinstance(image_input, Image.Image):
                # PIL Image - just convert to RGB
                image = image_input.convert('RGB')
            
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                image = self._load_from_array(image_input)
            
            elif isinstance(image_input, bytes):
                # Raw bytes
                image = self._load_from_bytes(image_input)
            
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Validate image if requested
            if validate:
                self._validate_image(image)
            
            return image
            
        except Exception as e:
            logging.error(f"Failed to load image: {e}")
            raise
    
    def _validate_image(self, image: Image.Image):
        """Validate image properties."""
        # Check image size
        if min(image.size) < 32:
            logging.warning(f"Very small image: {image.size}")
        
        if max(image.size) > 4096:
            logging.warning(f"Very large image: {image.size}, consider resizing")
        
        # Check aspect ratio
        aspect_ratio = max(image.size) / min(image.size)
        if aspect_ratio > 10:
            logging.warning(f"Extreme aspect ratio: {aspect_ratio:.2f}")
    
    def process(self, 
                image_input: Union[str, Path, Image.Image, np.ndarray, bytes],
                return_tensors: str = 'pt') -> torch.Tensor:
        """
        Process image into model-ready tensor format.
        
        Args:
            image_input: Image in various input formats
            return_tensors: Format for output tensors ('pt' for PyTorch)
            
        Returns:
            Processed image tensor with shape [1, C, H, W]
        """
        try:
            # Load and standardize image
            image = self.load_image(image_input)
            
            # Apply transformations
            tensor = self.transform(image)
            
            # Add batch dimension if not present
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            
            if return_tensors == 'pt':
                return tensor
            elif return_tensors == 'np':
                return tensor.numpy()
            else:
                raise ValueError(f"Unsupported return_tensors format: {return_tensors}")
                
        except Exception as e:
            logging.error(f"Failed to process image: {e}")
            raise
    
    def process_batch(self, 
                      image_inputs: List[Union[str, Path, Image.Image, np.ndarray, bytes]],
                      return_tensors: str = 'pt') -> torch.Tensor:
        """
        Process multiple images into a batched tensor.
        
        Args:
            image_inputs: List of images in various formats
            return_tensors: Format for output tensors
            
        Returns:
            Batched tensor with shape [B, C, H, W]
        """
        try:
            processed_images = []
            
            for i, image_input in enumerate(image_inputs):
                try:
                    tensor = self.process(image_input, return_tensors='pt')
                    processed_images.append(tensor)
                except Exception as e:
                    logging.warning(f"Failed to process image {i}: {e}, using zero tensor")
                    # Use zero tensor as placeholder
                    zero_tensor = torch.zeros(1, 3, *self.target_size)
                    processed_images.append(zero_tensor)
            
            # Stack into batch
            batch_tensor = torch.cat(processed_images, dim=0)
            
            if return_tensors == 'pt':
                return batch_tensor
            elif return_tensors == 'np':
                return batch_tensor.numpy()
            else:
                raise ValueError(f"Unsupported return_tensors format: {return_tensors}")
                
        except Exception as e:
            logging.error(f"Failed to process image batch: {e}")
            raise
    
    def get_info(self, image_input: Union[str, Path, Image.Image, np.ndarray, bytes]) -> dict:
        """
        Get information about an image without full processing.
        
        Args:
            image_input: Image in various formats
            
        Returns:
            Dictionary with image information
        """
        try:
            image = self.load_image(image_input, validate=False)
            
            return {
                'size': image.size,
                'mode': image.mode,
                'format': getattr(image, 'format', 'Unknown'),
                'aspect_ratio': image.size[0] / image.size[1],
                'megapixels': (image.size[0] * image.size[1]) / 1_000_000,
            }
        except Exception as e:
            return {'error': str(e)}


def create_clip_processor(model_name: str = "openai/clip-vit-base-patch32") -> ImageProcessor:
    """
    Create an ImageProcessor configured for CLIP models.
    
    Args:
        model_name: CLIP model name
        
    Returns:
        Configured ImageProcessor
    """
    # CLIP standard preprocessing
    return ImageProcessor(
        target_size=(224, 224),
        resize_mode='resize',
        normalize=True,
        mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
        std=[0.26862954, 0.26130258, 0.27577711]
    )


def create_processor_for_vision_tower(vision_tower_name: str) -> ImageProcessor:
    """
    Create an ImageProcessor configured for a specific vision tower.
    
    Args:
        vision_tower_name: Name of the vision tower model
        
    Returns:
        Configured ImageProcessor
    """
    if 'clip' in vision_tower_name.lower():
        return create_clip_processor(vision_tower_name)
    else:
        # Default to standard ImageNet preprocessing
        return ImageProcessor()