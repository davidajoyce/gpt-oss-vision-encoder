import json
import math
import os
from dataclasses import dataclass
from typing import Optional, List, Union

import torch
import torch.distributed as dist

from gpt_oss.torch.weights import Checkpoint
from gpt_oss.constants import (
    IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
    add_image_tokens, validate_image_tokens
)


@dataclass
class ModelConfig:
    num_hidden_layers: int = 36
    num_experts: int = 128
    experts_per_token: int = 4
    vocab_size: int = 201088
    hidden_size: int = 2880
    intermediate_size: int = 2880
    swiglu_limit: float = 7.0
    head_dim: int = 64
    num_attention_heads: int = 64
    num_key_value_heads: int = 8
    sliding_window: int = 128
    initial_context_length: int = 4096
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    
    # Vision-related configuration
    mm_vision_tower: str = None
    mm_projector_type: str = "linear"
    mm_hidden_size: int = None
    mm_vision_select_layer: int = -2
    mm_vision_select_feature: str = "patch"
    mm_patch_merge_type: str = "flat"
    use_mm_proj: bool = False
    tune_mm_mlp_adapter: bool = False
    freeze_mm_mlp_adapter: bool = False
    pretrain_mm_mlp_adapter: str = None


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    W = W[..., :-1]
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        assert config.intermediate_size % self.world_size == 0
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        t = torch.einsum("bec,be->bc", t, expert_weights)

        return x + t


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )
        
        # Vision components (initialized later if needed)
        self.vision_tower = None
        self.mm_projector = None

    def forward(self, x: torch.Tensor, images: Optional[torch.Tensor] = None) -> torch.Tensor:
        if images is not None and self.vision_tower is not None:
            return self.forward_multimodal(x, images)
        else:
            return self.forward_text_only(x)
    
    def forward_text_only(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x
    
    def forward_multimodal(self, input_ids: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        # Process images through vision tower
        image_features = self.vision_tower(images)
        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)
        
        # Process text tokens
        text_embeddings = self.embedding(input_ids)
        
        # For now, simple concatenation - this will be enhanced in Phase 2
        # In a full implementation, this would handle proper interleaving
        batch_size = text_embeddings.shape[0]
        if image_features.dim() == 3:  # [batch, num_patches, hidden_size]
            # Concatenate image features at the beginning
            combined_embeddings = torch.cat([image_features, text_embeddings], dim=1)
        else:
            # Handle single image case
            combined_embeddings = torch.cat([image_features.unsqueeze(0), text_embeddings], dim=1)
        
        # Forward through transformer blocks
        x = combined_embeddings
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x
    
    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args=None, fsdp=None):
        """Initialize vision components similar to LLaVA."""
        from .vision_tower import build_vision_tower
        from .vision_projector import build_vision_projector
        
        # Set vision tower configuration
        if hasattr(model_args, 'vision_tower'):
            self.config.mm_vision_tower = model_args.vision_tower
        
        # Build vision tower if needed
        if self.get_vision_tower() is None and self.config.mm_vision_tower:
            vision_tower = build_vision_tower(self.config)
            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            if vision_tower is not None:
                vision_tower.load_model()
        
        # Set up projector configuration
        if self.config.mm_vision_tower:
            self.config.use_mm_proj = True
            if hasattr(model_args, 'mm_projector_type'):
                self.config.mm_projector_type = model_args.mm_projector_type
            
            # Set hidden size from vision tower
            if self.vision_tower is not None:
                vision_tower = self.get_vision_tower()
                self.config.mm_hidden_size = vision_tower.hidden_size
        
        # Build projector if needed
        if getattr(self, 'mm_projector', None) is None and self.config.use_mm_proj:
            self.mm_projector = build_vision_projector(self.config)
        
        # Load pretrained projector weights if specified
        if hasattr(model_args, 'pretrain_mm_mlp_adapter') and model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            
            if self.mm_projector is not None:
                self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

    def initialize_image_tokenizer(self, tokenizer):
        """
        Initialize image tokens in the tokenizer and resize embeddings if needed.
        
        Args:
            tokenizer: Tokenizer instance to modify
            
        Returns:
            int: Image token ID
        """
        print("=== Initializing Image Tokenizer ===")
        
        # Get initial vocabulary size
        original_vocab_size = len(tokenizer)
        print(f"Original vocabulary size: {original_vocab_size}")
        
        # Add image tokens to vocabulary
        image_token_id = add_image_tokens(tokenizer)
        
        # Check if vocabulary size changed
        new_vocab_size = len(tokenizer)
        num_new_tokens = new_vocab_size - original_vocab_size
        
        print(f"New vocabulary size: {new_vocab_size}")
        print(f"Added {num_new_tokens} new tokens")
        
        # Always resize model embeddings to match tokenizer size
        # This ensures model can handle all tokens in the tokenizer vocabulary
        current_model_vocab_size = self.config.vocab_size
        print(f"Current model vocab size: {current_model_vocab_size}")
        
        if new_vocab_size != current_model_vocab_size:
            print(f"Resizing model embeddings to match tokenizer...")
            self.resize_token_embeddings(new_vocab_size)
            
            # If we added new tokens, initialize their embeddings
            if num_new_tokens > 0:
                # Initialize new token embeddings with average of existing embeddings
                input_embeddings = self.embedding.weight.data
                output_embeddings = self.unembedding.weight.data
                
                # Calculate average embeddings (excluding the new tokens)
                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
                
                # Initialize new tokens with average embeddings
                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg
                
                print(f"✅ Initialized {num_new_tokens} new token embeddings")
        
        # Store the image token ID for later use
        self._image_token_id = image_token_id
        
        # Validate the configuration
        if validate_image_tokens(tokenizer):
            print("✅ Image tokenizer initialization successful")
        else:
            print("❌ Image tokenizer initialization failed")
            
        return image_token_id
    
    def resize_token_embeddings(self, new_vocab_size):
        """
        Resize token embeddings to accommodate new vocabulary size
        
        Args:
            new_vocab_size: New vocabulary size
        """
        old_vocab_size = self.embedding.num_embeddings
        
        if new_vocab_size == old_vocab_size:
            return
            
        print(f"Resizing embeddings from {old_vocab_size} to {new_vocab_size}")
        
        # Create new embedding layers
        old_embedding = self.embedding
        old_unembedding = self.unembedding
        
        # Create new layers with expanded vocabulary
        self.embedding = torch.nn.Embedding(
            new_vocab_size, 
            self.config.hidden_size, 
            device=old_embedding.weight.device, 
            dtype=old_embedding.weight.dtype
        )
        
        self.unembedding = torch.nn.Linear(
            self.config.hidden_size,
            new_vocab_size,
            bias=False,
            device=old_unembedding.weight.device,
            dtype=old_unembedding.weight.dtype
        )
        
        # Copy old weights (handle both expansion and contraction)
        with torch.no_grad():
            copy_size = min(old_vocab_size, new_vocab_size)
            self.embedding.weight[:copy_size] = old_embedding.weight[:copy_size]
            self.unembedding.weight[:copy_size] = old_unembedding.weight[:copy_size]
        
        # Update config
        self.config.vocab_size = new_vocab_size
        
        print(f"✅ Embeddings resized to {new_vocab_size}")

    def embed_tokens(self, input_ids):
        """
        Convert token IDs to embeddings
        
        Args:
            input_ids: Token IDs tensor
            
        Returns:
            Embeddings tensor
        """
        return self.embedding(input_ids)
    
    def set_image_token_id(self, image_token_id):
        """
        Manually set the image token ID (useful for testing)
        
        Args:
            image_token_id: Token ID for image tokens
        """
        self._image_token_id = image_token_id
        print(f"Set image token ID to: {image_token_id}")
    
    def encode_images(self, images):
        """
        Encode images to feature representations using vision tower and projector
        
        Args:
            images: Image tensor [batch, 3, H, W]
            
        Returns:
            Image features: [batch, num_patches, hidden_dim]
        """
        if self.vision_tower is None:
            raise ValueError("Vision tower not initialized. Call initialize_vision_modules() first.")
        
        # Extract features using vision tower
        image_features = self.get_vision_tower()(images)
        
        # Project to language space using mm_projector
        if self.mm_projector is not None:
            image_features = self.mm_projector(image_features)
        
        return image_features
    
    def prepare_multimodal_inputs(self, input_ids, images=None, labels=None):
        """
        Replace <image> tokens in input_ids with actual image features.
        Based on LLaVA's prepare_inputs_labels_for_multimodal() function.
        
        Args:
            input_ids: [batch, seq_len] text tokens including <image> placeholders
            images: [batch, 3, 224, 224] or None
            labels: [batch, seq_len] or None for training
            
        Returns:
            dict with:
                inputs_embeds: [batch, new_seq_len, hidden_dim] combined embeddings
                labels: [batch, new_seq_len] updated labels for training (if provided)
                attention_mask: [batch, new_seq_len] attention mask
        """
        # Handle text-only case (backward compatibility)
        if images is None or self.vision_tower is None:
            text_embeds = self.embed_tokens(input_ids)
            batch_size, seq_len = input_ids.shape
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=input_ids.device)
            
            result = {
                "inputs_embeds": text_embeds,
                "attention_mask": attention_mask
            }
            if labels is not None:
                result["labels"] = labels
            return result
        
        # Encode images to features
        image_features = self.encode_images(images)  # [batch, num_patches, hidden_dim]
        
        # Process each sequence in the batch
        new_input_embeds = []
        new_labels = []
        
        for batch_idx in range(input_ids.shape[0]):
            cur_input_ids = input_ids[batch_idx]
            cur_labels = labels[batch_idx] if labels is not None else None
            
            # Find <image> token positions
            # Need to check for both IMAGE_TOKEN_INDEX and actual token ID from tokenizer
            image_token_id = None
            try:
                # Try to get the actual token ID from the vocabulary
                if hasattr(self, '_image_token_id'):
                    image_token_id = self._image_token_id
                else:
                    # This should be set during tokenizer initialization, but fallback to IMAGE_TOKEN_INDEX
                    image_token_id = IMAGE_TOKEN_INDEX
            except:
                image_token_id = IMAGE_TOKEN_INDEX
                
            image_token_positions = torch.where(cur_input_ids == image_token_id)[0]
            num_images = len(image_token_positions)
            
            if num_images == 0:
                # No images in this sequence - process as text only
                text_embeds = self.embed_tokens(cur_input_ids)
                new_input_embeds.append(text_embeds)
                if cur_labels is not None:
                    new_labels.append(cur_labels)
                continue
            
            # Split sequence around image token positions
            text_segments = []
            label_segments = []
            
            # Process segments between image tokens
            prev_pos = 0
            for img_pos in image_token_positions.tolist() + [len(cur_input_ids)]:
                # Get text segment before this image token (or end of sequence)
                if img_pos > prev_pos:
                    text_segments.append(cur_input_ids[prev_pos:img_pos])
                    if cur_labels is not None:
                        label_segments.append(cur_labels[prev_pos:img_pos])
                else:
                    # Empty segment
                    text_segments.append(torch.tensor([], dtype=cur_input_ids.dtype, device=cur_input_ids.device))
                    if cur_labels is not None:
                        label_segments.append(torch.tensor([], dtype=cur_labels.dtype, device=cur_labels.device))
                
                prev_pos = img_pos + 1  # Skip the image token itself
            
            # Build combined embedding sequence
            combined_embeds = []
            combined_labels = []
            
            for i in range(len(text_segments)):
                # Add text segment (if not empty)
                if len(text_segments[i]) > 0:
                    text_embeds = self.embed_tokens(text_segments[i])
                    combined_embeds.append(text_embeds)
                    if cur_labels is not None:
                        combined_labels.append(label_segments[i])
                
                # Add image features (except after the last text segment)
                if i < num_images:
                    # Get image features for this batch
                    cur_image_features = image_features[batch_idx]  # [num_patches, hidden_dim]
                    combined_embeds.append(cur_image_features)
                    
                    if cur_labels is not None:
                        # Image tokens should be ignored in loss computation
                        num_patches = cur_image_features.shape[0]
                        img_labels = torch.full(
                            (num_patches,), 
                            IGNORE_INDEX, 
                            device=cur_labels.device, 
                            dtype=cur_labels.dtype
                        )
                        combined_labels.append(img_labels)
            
            # Concatenate all segments for this sequence
            if len(combined_embeds) > 0:
                final_embeds = torch.cat(combined_embeds, dim=0)
                new_input_embeds.append(final_embeds)
                
                if cur_labels is not None and len(combined_labels) > 0:
                    final_labels = torch.cat(combined_labels, dim=0)
                    new_labels.append(final_labels)
        
        # Pad sequences to same length
        return self.pad_sequences(new_input_embeds, new_labels if labels is not None else None)
    
    def pad_sequences(self, input_embeds_list, labels_list=None):
        """
        Pad variable-length sequences to same length
        
        Args:
            input_embeds_list: List of embedding tensors with different lengths
            labels_list: List of label tensors (optional)
            
        Returns:
            dict with padded tensors
        """
        if len(input_embeds_list) == 0:
            raise ValueError("Empty input_embeds_list")
            
        # Calculate maximum length
        max_len = max(x.shape[0] for x in input_embeds_list)
        batch_size = len(input_embeds_list)
        hidden_dim = input_embeds_list[0].shape[1]
        device = input_embeds_list[0].device
        dtype = input_embeds_list[0].dtype
        
        # Create padded tensors
        padded_embeds = torch.zeros(batch_size, max_len, hidden_dim, dtype=dtype, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        
        padded_labels = None
        if labels_list is not None:
            padded_labels = torch.full(
                (batch_size, max_len), 
                IGNORE_INDEX, 
                dtype=labels_list[0].dtype, 
                device=device
            )
        
        # Fill in the actual data
        for i, embeds in enumerate(input_embeds_list):
            seq_len = embeds.shape[0]
            padded_embeds[i, :seq_len] = embeds
            attention_mask[i, :seq_len] = True
            
            if labels_list is not None and i < len(labels_list):
                padded_labels[i, :seq_len] = labels_list[i]
        
        # Build result dictionary
        result = {
            "inputs_embeds": padded_embeds,
            "attention_mask": attention_mask
        }
        
        if padded_labels is not None:
            result["labels"] = padded_labels
        
        return result

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        if not isinstance(device, torch.device):
            device = torch.device(device)

        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load weights
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        checkpoint = Checkpoint(path, device)

        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Note: it would be more efficient to do sharding before upcasting from MXFP4,
            # but for simplicity we do it after.
            if "mlp1" in name:  # both weight and bias
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2
                    * per_rank_intermediate_size : (my_rank + 1) * 2
                    * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # only weight
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank
                    * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                ]
            try:
                param.data.copy_(loaded_tensor)
            except:
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            if predicted_token in stop_tokens:
                break
