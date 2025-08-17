import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, List, Union

try:
    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class VisionTower(nn.Module, ABC):
    """Abstract base class for vision towers."""
    
    def __init__(self, vision_tower_name: str, args=None, delay_load: bool = False):
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1) if args else -1
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch') if args else 'patch'
        
        if not delay_load:
            self.load_model()
    
    @abstractmethod
    def load_model(self):
        pass
    
    @abstractmethod
    def forward(self, images):
        pass
    
    @property
    @abstractmethod
    def dummy_feature(self):
        pass
    
    @property
    @abstractmethod
    def dtype(self):
        pass
    
    @property
    @abstractmethod
    def device(self):
        pass
    
    @property
    @abstractmethod
    def config(self):
        pass
    
    @property
    @abstractmethod
    def hidden_size(self):
        pass
    
    @property
    @abstractmethod
    def num_patches_per_side(self):
        pass
    
    @property
    @abstractmethod
    def num_patches(self):
        pass


class CLIPVisionTower(VisionTower):
    """CLIP-based vision tower implementation."""
    
    def __init__(self, vision_tower_name: str, args=None, delay_load: bool = False):
        super().__init__(vision_tower_name, args, delay_load)
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for CLIPVisionTower")
    
    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f'{self.vision_tower_name} is already loaded, `load_model` called again, skipping.')
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)
        
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]  # Remove CLS token
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2


def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Build vision tower based on the configuration.
    
    Args:
        vision_tower_cfg: Either a string (model name) or config object
        **kwargs: Additional arguments
        
    Returns:
        VisionTower: The vision tower instance
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = isinstance(vision_tower, str) and vision_tower.startswith('/')
    
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "clip" in vision_tower.lower():
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')