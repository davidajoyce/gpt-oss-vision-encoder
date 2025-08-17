# Image Token Constants for Multimodal Processing
# Based on LLaVA's approach for vision-language integration

# Core multimodal constants
IGNORE_INDEX = -100                    # For masking loss on image tokens during training
IMAGE_TOKEN_INDEX = -200               # Special token ID for image placeholders
DEFAULT_IMAGE_TOKEN = "<image>"        # Text placeholder for images in sequences

# Additional image tokens for fine-grained control (following LLaVA)
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"

def add_image_tokens(tokenizer):
    """
    Add image tokens to tokenizer vocabulary
    
    Args:
        tokenizer: Hugging Face tokenizer instance
        
    Returns:
        int: Token ID for the image token
    """
    # Add the primary image token
    num_added_tokens = tokenizer.add_special_tokens({
        "additional_special_tokens": [DEFAULT_IMAGE_TOKEN]
    })
    
    # Get the token ID
    image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    
    print(f"Added {num_added_tokens} image token(s) to vocabulary")
    print(f"Image token '{DEFAULT_IMAGE_TOKEN}' has ID: {image_token_id}")
    
    return image_token_id

def get_image_token_id(tokenizer):
    """
    Get the token ID for the image token
    
    Args:
        tokenizer: Tokenizer instance
        
    Returns:
        int: Token ID for image token, or IMAGE_TOKEN_INDEX if not in vocab
    """
    try:
        return tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
    except:
        # Fallback to our fixed index
        return IMAGE_TOKEN_INDEX

def validate_image_tokens(tokenizer):
    """
    Validate that image tokens are properly configured
    
    Args:
        tokenizer: Tokenizer instance
        
    Returns:
        bool: True if properly configured
    """
    try:
        # Check if image token exists
        image_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        
        # Check if it's a valid token (not UNK)
        if image_token_id == tokenizer.unk_token_id:
            print(f"WARNING: Image token '{DEFAULT_IMAGE_TOKEN}' maps to UNK token")
            return False
            
        # Check if we can decode it back
        decoded = tokenizer.decode([image_token_id])
        if DEFAULT_IMAGE_TOKEN not in decoded:
            print(f"WARNING: Image token ID {image_token_id} doesn't decode properly")
            return False
            
        print(f"✅ Image token validation passed: '{DEFAULT_IMAGE_TOKEN}' -> {image_token_id}")
        return True
        
    except Exception as e:
        print(f"❌ Image token validation failed: {e}")
        return False