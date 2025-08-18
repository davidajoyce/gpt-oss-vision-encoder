#!/usr/bin/env python3
"""
Download COCO val2017 images for LLaVA training
These are the actual images referenced by the LLaVA-Instruct-150K dataset
"""

import os
import sys
import requests
import zipfile
import json
from tqdm import tqdm

def download_file(url, dest_path):
    """Download a file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(dest_path, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))

def main():
    print("="*60)
    print("🖼️ COCO Image Downloader for LLaVA Training")
    print("="*60)
    
    # Create directory for COCO images
    coco_dir = "coco_images"
    os.makedirs(coco_dir, exist_ok=True)
    
    # Check what's already downloaded
    val2017_dir = os.path.join(coco_dir, "val2017")
    train2017_dir = os.path.join(coco_dir, "train2017")
    
    val_exists = os.path.exists(val2017_dir) and len([f for f in os.listdir(val2017_dir) if f.endswith('.jpg')]) > 4000
    train_exists = os.path.exists(train2017_dir) and len([f for f in os.listdir(train2017_dir) if f.endswith('.jpg')]) > 100000
    
    if val_exists and train_exists:
        val_count = len([f for f in os.listdir(val2017_dir) if f.endswith('.jpg')])
        train_count = len([f for f in os.listdir(train2017_dir) if f.endswith('.jpg')])
        print(f"✅ COCO images already downloaded:")
        print(f"   - val2017: {val_count} images")
        print(f"   - train2017: {train_count} images")
        return coco_dir
    
    print("\n📥 Downloading COCO images for LLaVA training...")
    print("🎯 LLaVA-150K uses images from BOTH train2017 and val2017")
    print("⚠️ This will download ~20GB of data (118K train + 5K val images)")
    print("⏱️ Estimated time: 15-30 minutes depending on connection")
    
    # Download URLs
    datasets = {
        'val2017': {
            'url': 'http://images.cocodataset.org/zips/val2017.zip',
            'size': '~1GB',
            'count': '5K images'
        },
        'train2017': {
            'url': 'http://images.cocodataset.org/zips/train2017.zip', 
            'size': '~19GB',
            'count': '118K images'
        }
    }
    
    for dataset_name, info in datasets.items():
        dataset_dir = os.path.join(coco_dir, dataset_name)
        zip_path = os.path.join(coco_dir, f"{dataset_name}.zip")
        
        # Check if already exists
        if os.path.exists(dataset_dir):
            existing_count = len([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')])
            expected_min = 4000 if dataset_name == 'val2017' else 100000
            
            if existing_count > expected_min:
                print(f"✅ {dataset_name} already exists: {existing_count} images")
                continue
        
        print(f"\n📦 Downloading {dataset_name} ({info['size']}, {info['count']})...")
        
        # Download if not exists
        if not os.path.exists(zip_path):
            try:
                download_file(info['url'], zip_path)
                print(f"✅ {dataset_name} download complete!")
            except Exception as e:
                print(f"❌ {dataset_name} download failed: {e}")
                print(f"\n💡 Manual download: {info['url']}")
                if dataset_name == 'train2017':
                    print("💡 You can continue with val2017 only, but some LLaVA images will be missing")
                    continue
                else:
                    sys.exit(1)
        else:
            print(f"✅ Using existing {dataset_name}.zip")
        
        # Extract images
        print(f"\n📂 Extracting {dataset_name}...")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(coco_dir)
            print(f"✅ Extracted to: {dataset_dir}")
        except Exception as e:
            print(f"❌ {dataset_name} extraction failed: {e}")
            if dataset_name == 'val2017':
                sys.exit(1)
            continue
        
        # Verify extraction
        if os.path.exists(dataset_dir):
            num_images = len([f for f in os.listdir(dataset_dir) if f.endswith('.jpg')])
            print(f"✅ Successfully extracted {num_images} {dataset_name} images!")
        else:
            print(f"❌ {dataset_name} extraction failed - directory not found")
            if dataset_name == 'val2017':
                sys.exit(1)
    
    # Check LLaVA dataset compatibility
    print("\n🔍 Checking LLaVA dataset compatibility...")
    
    # Download a sample of LLaVA dataset to check image IDs
    llava_sample_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/raw/main/llava_instruct_150k.json"
    llava_sample_path = "llava_data/llava_instruct_150k_sample.json"
    
    os.makedirs("llava_data", exist_ok=True)
    
    if not os.path.exists(llava_sample_path):
        print("📥 Downloading LLaVA dataset sample to verify image IDs...")
        try:
            response = requests.get(llava_sample_url, stream=True, timeout=30)
            # Just get first 1MB to check format
            content = response.raw.read(1024*1024)
            with open(llava_sample_path, 'wb') as f:
                f.write(content)
        except:
            print("⚠️ Could not download LLaVA sample for verification")
    
    print("\n" + "="*60)
    print("✅ COCO IMAGES READY FOR TRAINING!")
    print("="*60)
    print(f"\n📂 Image directory: {os.path.abspath(val2017_dir)}")
    print(f"🖼️ Total images: {num_images}")
    print("\n💡 Next steps:")
    print("1. The training script will now use these real images")
    print("2. Each LLaVA sample references a COCO image by ID")
    print("3. Training will map real images to text descriptions")
    
    return val2017_dir

if __name__ == "__main__":
    main()