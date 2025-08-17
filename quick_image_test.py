#!/usr/bin/env python3
"""
Quick test to see what the model actually predicts for images
"""

import sys
import os
sys.path.append('.')

import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from transformers import CLIPVisionModel, CLIPImageProcessor, AutoTokenizer
import argparse

def load_model(checkpoint_path):
    """Load the trained model"""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    from gpt_oss.torch.model import Transformer, ModelConfig
    
    # Load checkpoint 
    try:
        with torch.serialization.safe_globals([ModelConfig]):
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    except:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model_config = checkpoint['model_config']
    
    # Create model
    model = Transformer(model_config, device='cpu').float()
    
    # Separate model and projector weights
    model_state_dict = checkpoint['model_state_dict']
    clean_model_state = {}
    projector_state = {}
    
    for key, value in model_state_dict.items():
        if key.startswith('mm_projector.'):
            projector_key = key.replace('mm_projector.', '')
            projector_state[projector_key] = value
        else:
            clean_model_state[key] = value
    
    model.load_state_dict(clean_model_state)
    
    # Create projector
    projector = nn.Linear(768, model_config.hidden_size).float()
    if projector_state:
        projector.load_state_dict(projector_state)
    elif 'projector_state_dict' in checkpoint:
        projector.load_state_dict(checkpoint['projector_state_dict'])
    
    return model, projector

def test_image_prediction(model, projector, image_path=None):
    """Test what the model predicts for an image"""
    print(f"\n🔍 Testing image prediction...")
    
    # Setup components
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def vision_tower_wrapper(images):
        with torch.no_grad():
            return vision_tower(images).last_hidden_state
    
    model.vision_tower = vision_tower_wrapper
    model.mm_projector = projector
    
    # Initialize image tokenizer
    try:
        image_token_id = model.initialize_image_tokenizer(tokenizer)
        model.set_image_token_id(image_token_id)
    except:
        pass
    
    # Create or load test image
    if image_path and os.path.exists(image_path):
        print(f"📷 Loading: {image_path}")
        image = Image.open(image_path).convert('RGB')
    else:
        print(f"🎨 Creating test image: Red Circle")
        image = Image.new('RGB', (224, 224), 'white')
        draw = ImageDraw.Draw(image)
        draw.ellipse([50, 50, 174, 174], fill='red')
    
    from gpt_oss.constants import DEFAULT_IMAGE_TOKEN
    
    # Test questions
    questions = [
        f"What is {DEFAULT_IMAGE_TOKEN}?",
        f"Describe {DEFAULT_IMAGE_TOKEN}.",
        f"What color is {DEFAULT_IMAGE_TOKEN}?",
        f"What shape is {DEFAULT_IMAGE_TOKEN}?"
    ]
    
    model.eval()
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        
        try:
            # Process image
            pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values']
            
            # Tokenize question
            input_ids = tokenizer.encode(question, return_tensors="pt")
            
            # Prepare multimodal inputs
            with torch.no_grad():
                multimodal_inputs = model.prepare_multimodal_inputs(
                    input_ids=input_ids,
                    images=pixel_values
                )
                
                embeddings = multimodal_inputs["inputs_embeds"].float()
                
                print(f"  Input tokens: {tokenizer.decode(input_ids[0])}")
                print(f"  Input shape: {embeddings.shape}")
                
                # Generate next tokens with actual decoding
                generated_tokens = []
                current_embeddings = embeddings
                
                for step in range(10):  # Generate up to 10 tokens
                    # Forward pass
                    hidden = model.norm(current_embeddings)
                    logits = model.unembedding(hidden).float()
                    
                    # Get probabilities for last position
                    last_logits = logits[0, -1, :]
                    probs = torch.softmax(last_logits, dim=0)
                    
                    # Get top token
                    top_token_id = torch.argmax(probs).item()
                    top_prob = probs[top_token_id].item()
                    
                    # Decode token
                    token_text = tokenizer.decode([top_token_id])
                    
                    print(f"    Step {step+1}: '{token_text}' (prob: {top_prob:.3f})")
                    
                    generated_tokens.append(token_text)
                    
                    # Stop if EOS
                    if top_token_id == tokenizer.eos_token_id:
                        print(f"    → EOS reached")
                        break
                    
                    # Add token to sequence for next iteration
                    next_embed = model.embed_tokens(torch.tensor([[top_token_id]])).float()
                    current_embeddings = torch.cat([current_embeddings, next_embed], dim=1)
                
                # Show full generated response
                full_response = "".join(generated_tokens)
                print(f"  🤖 Generated: '{full_response}'")
                
                # Show top 5 alternatives for first token
                top5 = torch.topk(probs, 5)
                print(f"  📊 Top 5 first tokens:")
                for i, (prob, token_id) in enumerate(zip(top5.values, top5.indices)):
                    token = tokenizer.decode([token_id.item()])
                    print(f"    {i+1}. '{token}' ({prob.item():.3f})")
                    
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to model file")
    parser.add_argument("--image", help="Path to test image")
    args = parser.parse_args()
    
    model, projector = load_model(args.model_path)
    test_image_prediction(model, projector, args.image)

if __name__ == "__main__":
    main()