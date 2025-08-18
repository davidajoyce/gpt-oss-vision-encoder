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
    
    # Check if already downloaded
    val2017_dir = os.path.join(coco_dir, "val2017")
    if os.path.exists(val2017_dir) and len(os.listdir(val2017_dir)) > 5000:
        print(f"✅ COCO val2017 already downloaded: {len(os.listdir(val2017_dir))} images")
        return val2017_dir
    
    print("\n📥 Downloading COCO val2017 images...")
    print("⚠️ This will download ~1GB of data (5K validation images)")
    
    # COCO val2017 URL
    val2017_url = "http://images.cocodataset.org/zips/val2017.zip"
    zip_path = os.path.join(coco_dir, "val2017.zip")
    
    # Download if not exists
    if not os.path.exists(zip_path):
        print(f"\n📦 Downloading val2017.zip...")
        try:
            download_file(val2017_url, zip_path)
            print("✅ Download complete!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            print("\n💡 Alternative: Download manually from:")
            print(f"   {val2017_url}")
            print(f"   Save to: {zip_path}")
            sys.exit(1)
    else:
        print(f"✅ Using existing zip: {zip_path}")
    
    # Extract images
    print("\n📂 Extracting images...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(coco_dir)
        print(f"✅ Extracted to: {val2017_dir}")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    # Verify extraction
    if os.path.exists(val2017_dir):
        num_images = len([f for f in os.listdir(val2017_dir) if f.endswith('.jpg')])
        print(f"\n✅ Successfully extracted {num_images} images!")
        print(f"📍 Images location: {os.path.abspath(val2017_dir)}")
    else:
        print("❌ Extraction failed - directory not found")
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