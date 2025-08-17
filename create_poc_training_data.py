#!/usr/bin/env python3
"""
Create synthetic training data for proof-of-concept multimodal training.

This script generates:
1. Small synthetic images (to avoid data downloading)
2. Simple image-caption pairs in LLaVA format
3. Minimal dataset for Stage 1 projector training

Usage:
    python create_poc_training_data.py
"""

import os
import json
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

def create_synthetic_images(output_dir: str, num_images: int = 100) -> list:
    """Create synthetic images with simple patterns and shapes."""
    
    images_dir = Path(output_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = []
    
    # Define simple patterns and colors
    colors = [
        ('red', (255, 100, 100)),
        ('blue', (100, 100, 255)), 
        ('green', (100, 255, 100)),
        ('yellow', (255, 255, 100)),
        ('purple', (255, 100, 255)),
        ('orange', (255, 180, 100)),
    ]
    
    shapes = ['circle', 'square', 'triangle']
    
    for i in range(num_images):
        # Create 224x224 image (standard vision model input size)
        img = Image.new('RGB', (224, 224), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Pick random color and shape
        color_name, color_rgb = random.choice(colors)
        shape = random.choice(shapes)
        
        # Draw shape in center
        center_x, center_y = 112, 112
        size = random.randint(40, 80)
        
        if shape == 'circle':
            bbox = [center_x - size, center_y - size, center_x + size, center_y + size]
            draw.ellipse(bbox, fill=color_rgb)
        elif shape == 'square':
            bbox = [center_x - size, center_y - size, center_x + size, center_y + size]
            draw.rectangle(bbox, fill=color_rgb)
        elif shape == 'triangle':
            points = [
                (center_x, center_y - size),  # top
                (center_x - size, center_y + size),  # bottom left
                (center_x + size, center_y + size),  # bottom right
            ]
            draw.polygon(points, fill=color_rgb)
        
        # Add some random details
        if random.random() < 0.3:  # 30% chance of border
            if shape == 'circle':
                draw.ellipse(bbox, outline=(0, 0, 0), width=3)
            else:
                draw.rectangle(bbox, outline=(0, 0, 0), width=3)
        
        # Save image
        image_name = f"synthetic_{i:04d}.jpg"
        image_path = images_dir / image_name
        img.save(image_path, quality=95)
        image_paths.append(image_name)
        
        if i % 20 == 0:
            print(f"Created {i}/{num_images} synthetic images...")
    
    print(f"✅ Created {num_images} synthetic images in {images_dir}")
    return image_paths

def create_training_conversations(image_paths: list) -> list:
    """Create simple training conversations for each image."""
    
    conversations = []
    
    # Simple templates for captions
    templates = [
        "This image shows a {color} {shape}.",
        "I can see a {color} {shape} in the image.",
        "The image contains a {color} {shape}.",
        "There is a {color} {shape} displayed.",
        "A {color} {shape} is visible in this image.",
    ]
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['circle', 'square', 'triangle']
    
    for i, image_path in enumerate(image_paths):
        # Generate simple caption
        color = random.choice(colors)
        shape = random.choice(shapes)
        template = random.choice(templates)
        caption = template.format(color=color, shape=shape)
        
        # Create conversation in LLaVA format
        conversation = {
            "id": f"synthetic_{i:04d}",
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nDescribe this image."
                },
                {
                    "from": "gpt", 
                    "value": caption
                }
            ]
        }
        
        conversations.append(conversation)
    
    return conversations

def create_stage1_data(output_dir: str, num_samples: int = 100):
    """Create Stage 1 training data (image-caption alignment)."""
    
    print(f"🎯 Creating Stage 1 POC training data with {num_samples} samples...")
    
    # Create synthetic images
    image_paths = create_synthetic_images(output_dir, num_samples)
    
    # Create conversations
    conversations = create_training_conversations(image_paths)
    
    # Save training data in LLaVA format
    data_file = Path(output_dir) / "stage1_poc_data.json"
    with open(data_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"✅ Saved {len(conversations)} conversations to {data_file}")
    
    # Create simple validation data (subset)
    val_conversations = conversations[:min(10, len(conversations))]
    val_file = Path(output_dir) / "stage1_poc_val.json"
    with open(val_file, 'w') as f:
        json.dump(val_conversations, f, indent=2)
    
    print(f"✅ Saved {len(val_conversations)} validation samples to {val_file}")
    
    return data_file, val_file

def create_stage2_data(output_dir: str, num_samples: int = 50):
    """Create Stage 2 training data (instruction following)."""
    
    print(f"🎯 Creating Stage 2 POC training data with {num_samples} samples...")
    
    # Use same images but with more complex conversations
    image_paths = [f"synthetic_{i:04d}.jpg" for i in range(num_samples)]
    
    conversations = []
    
    # More complex instruction templates
    instruction_templates = [
        {
            "question": "<image>\nWhat color is the main object in this image?",
            "answers": ["red", "blue", "green", "yellow", "purple", "orange"]
        },
        {
            "question": "<image>\nWhat shape do you see in the image?",
            "answers": ["circle", "square", "triangle"]
        },
        {
            "question": "<image>\nDescribe the image in detail.",
            "answers": ["This image shows a {color} {shape} on a light gray background."]
        },
        {
            "question": "<image>\nIs there a geometric shape in this image?",
            "answers": ["Yes, there is a {shape} in the image."]
        }
    ]
    
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
    shapes = ['circle', 'square', 'triangle']
    
    for i, image_path in enumerate(image_paths):
        template = random.choice(instruction_templates)
        color = random.choice(colors)
        shape = random.choice(shapes)
        
        # Format answer if needed
        if isinstance(template["answers"], list) and len(template["answers"]) == 1:
            answer = template["answers"][0].format(color=color, shape=shape)
        else:
            answer = random.choice(template["answers"])
        
        conversation = {
            "id": f"instruction_{i:04d}",
            "image": image_path,
            "conversations": [
                {
                    "from": "human",
                    "value": template["question"]
                },
                {
                    "from": "gpt",
                    "value": answer
                }
            ]
        }
        
        conversations.append(conversation)
    
    # Save Stage 2 data
    data_file = Path(output_dir) / "stage2_poc_data.json"
    with open(data_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    
    print(f"✅ Saved {len(conversations)} instruction conversations to {data_file}")
    return data_file

def main():
    """Create complete POC training dataset."""
    
    # Create output directory
    output_dir = "./data/poc_training"
    os.makedirs(output_dir, exist_ok=True)
    
    print("🚀 Creating Proof-of-Concept Training Data")
    print("=" * 50)
    
    # Create Stage 1 data (more samples for alignment)
    stage1_data, stage1_val = create_stage1_data(output_dir, num_samples=100)
    
    # Create Stage 2 data (fewer samples for instruction following)
    stage2_data = create_stage2_data(output_dir, num_samples=50)
    
    print("\n📊 Dataset Summary:")
    print(f"📁 Data directory: {output_dir}")
    print(f"🖼️  Images: 100 synthetic images (224x224 pixels)")
    print(f"📝 Stage 1 training: 100 image-caption pairs")
    print(f"🔍 Stage 1 validation: 10 samples")
    print(f"🎯 Stage 2 training: 50 instruction-following pairs")
    
    print("\n🎯 Next Steps:")
    print("1. Run Stage 1 training: python train_poc_stage1.py")
    print("2. Validate projector: python test_trained_projector.py")
    print("3. Run Stage 2 training: python train_poc_stage2.py")
    
    print("\n✅ POC training data creation complete!")

if __name__ == "__main__":
    main()