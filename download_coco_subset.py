#!/usr/bin/env python3
"""
Download a SUBSET of COCO train2017 images for testing
This is much faster than downloading the full 19GB train2017 dataset
"""

import os
import sys
import requests
import json
from tqdm import tqdm
from datasets import load_dataset
import glob

def main():
    print("="*60)
    print("🎯 COCO SUBSET DOWNLOADER - For Testing LLaVA")
    print("="*60)
    
    # Configuration
    MAX_TRAIN_IMAGES = 10000  # Download only first 10K train images (~1.5GB)
    coco_dir = "coco_images"
    os.makedirs(coco_dir, exist_ok=True)
    
    val2017_dir = os.path.join(coco_dir, "val2017")
    train2017_dir = os.path.join(coco_dir, "train2017")
    
    # Check val2017 (always download - small and essential)
    if not os.path.exists(val2017_dir) or len(glob.glob(os.path.join(val2017_dir, "*.jpg"))) < 4000:
        print("\n📦 Step 1: Downloading COCO val2017 (1GB, essential)...")
        os.system("python download_coco_images.py")
    else:
        print("✅ val2017 already exists")
    
    # Check what LLaVA images we actually need
    print(f"\n🔍 Step 2: Analyzing LLaVA dataset to find needed train2017 images...")
    
    needed_train_images = set()
    
    try:
        print("📥 Loading LLaVA-150K dataset to check image requirements...")
        
        # Try to load a sample of LLaVA dataset
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train", streaming=True)
        
        print("📊 Scanning dataset for train2017 image IDs...")
        sample_count = 0
        train_image_count = 0
        
        for item in dataset:
            sample_count += 1
            if sample_count > 20000:  # Check first 20K samples
                break
                
            if sample_count % 5000 == 0:
                print(f"   Scanned {sample_count} samples, found {train_image_count} train2017 images needed")
            
            # Check if this sample needs a train2017 image
            if 'image' in item:
                image_id = item['image']
                if not image_id.endswith('.jpg'):
                    image_id = f"{image_id}.jpg"
                
                # Check if it exists in val2017 (if not, probably train2017)
                val_path = os.path.join(val2017_dir, image_id)
                if not os.path.exists(val_path):
                    needed_train_images.add(image_id)
                    train_image_count += 1
                    
                    # Limit how many we'll download
                    if len(needed_train_images) >= MAX_TRAIN_IMAGES:
                        break
        
        print(f"\n📊 Analysis complete:")
        print(f"   - Samples analyzed: {sample_count}")
        print(f"   - train2017 images needed: {len(needed_train_images)}")
        print(f"   - Will download: {min(len(needed_train_images), MAX_TRAIN_IMAGES)} images")
        
    except Exception as e:
        print(f"⚠️ Could not analyze LLaVA dataset: {e}")
        print("💡 Will download first 10K train2017 images as fallback")
        
        # Fallback: create list of common COCO train2017 IDs
        needed_train_images = {f"{i:012d}.jpg" for i in range(1, MAX_TRAIN_IMAGES + 1)}
    
    # Create train2017 directory
    os.makedirs(train2017_dir, exist_ok=True)
    
    # Check what we already have
    existing_images = set(os.path.basename(f) for f in glob.glob(os.path.join(train2017_dir, "*.jpg")))
    images_to_download = needed_train_images - existing_images
    
    if len(existing_images) > 0:
        print(f"✅ Already have {len(existing_images)} train2017 images")
    
    if len(images_to_download) == 0:
        print("✅ All needed train2017 images already downloaded!")
        print_summary(val2017_dir, train2017_dir)
        return
    
    print(f"\n📦 Step 3: Downloading {len(images_to_download)} train2017 images...")
    print(f"💾 Estimated size: ~{len(images_to_download) * 0.15:.1f}MB")
    
    # Download individual images
    base_url = "http://images.cocodataset.org/train2017"
    downloaded = 0
    failed = 0
    
    for image_id in tqdm(images_to_download, desc="Downloading images"):
        image_url = f"{base_url}/{image_id}"
        image_path = os.path.join(train2017_dir, image_id)
        
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                downloaded += 1
            else:
                failed += 1
                if failed < 10:  # Only show first 10 failures
                    print(f"⚠️ Failed to download {image_id}: {response.status_code}")
        except Exception as e:
            failed += 1
            if failed < 10:
                print(f"⚠️ Error downloading {image_id}: {e}")
    
    print(f"\n📊 Download Results:")
    print(f"   ✅ Successfully downloaded: {downloaded} images")
    if failed > 0:
        print(f"   ⚠️ Failed downloads: {failed} images")
    
    print_summary(val2017_dir, train2017_dir)

def print_summary(val2017_dir, train2017_dir):
    val_count = len(glob.glob(os.path.join(val2017_dir, "*.jpg"))) if os.path.exists(val2017_dir) else 0
    train_count = len(glob.glob(os.path.join(train2017_dir, "*.jpg"))) if os.path.exists(train2017_dir) else 0
    
    print("\n" + "="*60)
    print("✅ COCO SUBSET READY FOR LLAVA TRAINING!")
    print("="*60)
    print(f"\n📊 Final Dataset:")
    print(f"   - val2017: {val_count} images")
    print(f"   - train2017: {train_count} images")
    print(f"   - Total: {val_count + train_count} images")
    
    print(f"\n💾 Storage Used:")
    if val_count > 0:
        print(f"   - val2017: ~1GB")
    if train_count > 0:
        print(f"   - train2017: ~{train_count * 0.15:.0f}MB")
    
    print(f"\n🎯 This subset should cover most LLaVA-150K images!")
    print(f"   - Much faster than downloading full 19GB train2017")
    print(f"   - Should eliminate most 'Failed to load' errors")
    print(f"   - Perfect for testing the training pipeline")
    
    print(f"\n🚀 Ready to train:")
    print(f"   python train_llava_fixed.py")

if __name__ == "__main__":
    main()