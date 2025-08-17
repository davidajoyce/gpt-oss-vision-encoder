import torch
import torch.nn as nn
import re
from typing import Optional


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    """
    Build vision projector that maps vision features to language model space.
    
    Args:
        config: ModelConfig with vision parameters
        delay_load: Whether to delay loading (for compatibility)
        
    Returns:
        nn.Module: The vision projector
    """
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    
    if not hasattr(config, 'mm_hidden_size') or config.mm_hidden_size is None:
        raise ValueError("mm_hidden_size must be set in config for vision projector")
    
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        if config.mm_hidden_size != config.hidden_size:
            raise ValueError(f"Identity projector requires mm_hidden_size ({config.mm_hidden_size}) "
                           f"== hidden_size ({config.hidden_size})")
        return IdentityMap()
    
    # Support for ResBlock-based projectors
    if projector_type.startswith('resblock'):
        depth_match = re.match(r'^resblock(\d+)$', projector_type)
        if depth_match:
            depth = int(depth_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(depth):
                modules.append(SimpleResBlock(config.hidden_size))
            return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')