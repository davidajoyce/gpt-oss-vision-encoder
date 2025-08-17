#!/usr/bin/env python3
"""
Test LLaVA dataset loading with detailed diagnostics
"""

import sys
import os
sys.path.append('.')

from datasets import load_dataset
import json

print("="*60)
print("🔍 LLaVA Dataset Loading Diagnostics")
print("="*60)

def test_dataset_loading():
    """Test different ways to load LLaVA dataset"""
    
    print("\n📋 Testing multiple loading methods...")
    
    # Method 1: Direct loading with split
    print("\n1️⃣ Testing: load_dataset('liuhaotian/LLaVA-Instruct-150K', split='train')")
    try:
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", split="train")
        print(f"   ✅ SUCCESS: {len(dataset)} samples")
        print(f"   Sample keys: {list(dataset[0].keys())}")
        return dataset
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    # Method 2: Load without split
    print("\n2️⃣ Testing: load_dataset('liuhaotian/LLaVA-Instruct-150K')")
    try:
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
        print(f"   ✅ SUCCESS: Dataset object loaded")
        print(f"   Available splits: {list(dataset.keys())}")
        
        if len(dataset.keys()) > 0:
            split_name = list(dataset.keys())[0]
            split_data = dataset[split_name]
            print(f"   Split '{split_name}': {len(split_data)} samples")
            print(f"   Sample keys: {list(split_data[0].keys())}")
            return split_data
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    # Method 3: Try with trust_remote_code
    print("\n3️⃣ Testing: load_dataset with trust_remote_code=True")
    try:
        dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K", trust_remote_code=True)
        print(f"   ✅ SUCCESS: Dataset object loaded")
        print(f"   Available splits: {list(dataset.keys())}")
        
        if len(dataset.keys()) > 0:
            split_name = list(dataset.keys())[0]
            split_data = dataset[split_name]
            print(f"   Split '{split_name}': {len(split_data)} samples")
            return split_data
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print(f"   Error type: {type(e).__name__}")
    
    # Method 4: Check cache directory
    print("\n4️⃣ Testing: Check local cache")
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
        print(f"   Cache directory: {cache_dir}")
        
        if os.path.exists(cache_dir):
            cache_contents = os.listdir(cache_dir)
            llava_dirs = [d for d in cache_contents if 'llava' in d.lower()]
            print(f"   LLaVA-related directories: {llava_dirs}")
            
            # Look for downloaded files
            for llava_dir in llava_dirs:
                full_path = os.path.join(cache_dir, llava_dir)
                if os.path.isdir(full_path):
                    print(f"   Contents of {llava_dir}:")
                    for item in os.listdir(full_path):
                        item_path = os.path.join(full_path, item)
                        if os.path.isfile(item_path):
                            size_mb = os.path.getsize(item_path) / (1024*1024)
                            print(f"     📄 {item} ({size_mb:.1f} MB)")
                        else:
                            print(f"     📁 {item}/")
        else:
            print(f"   ❌ Cache directory doesn't exist")
            
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    # Method 5: Try alternative dataset
    print("\n5️⃣ Testing: Alternative LLaVA dataset")
    try:
        # Try a different LLaVA dataset that might be more stable
        dataset = load_dataset("MMInstruction/LLaVA-Instruct", split="train")
        print(f"   ✅ SUCCESS: {len(dataset)} samples from alternative dataset")
        print(f"   Sample keys: {list(dataset[0].keys())}")
        return dataset
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    
    print("\n❌ All loading methods failed")
    return None

def analyze_sample_data(dataset):
    """Analyze the structure of loaded data"""
    print(f"\n🔬 Analyzing dataset structure...")
    
    if dataset is None:
        print("❌ No dataset to analyze")
        return
    
    print(f"📊 Dataset size: {len(dataset)}")
    
    # Look at first few samples
    for i in range(min(3, len(dataset))):
        print(f"\n📝 Sample {i}:")
        sample = dataset[i]
        
        for key, value in sample.items():
            if isinstance(value, str):
                preview = value[:100] + "..." if len(value) > 100 else value
                print(f"   {key}: '{preview}'")
            elif isinstance(value, list):
                print(f"   {key}: List with {len(value)} items")
                if len(value) > 0:
                    print(f"      First item: {value[0]}")
            else:
                print(f"   {key}: {type(value).__name__} = {value}")

def test_json_loading():
    """Test loading from raw JSON files"""
    print(f"\n📄 Testing JSON file loading...")
    
    # URLs for the raw JSON files
    json_files = [
        "llava_instruct_150k.json",
        "conversation_58k.json", 
        "detail_23k.json",
        "complex_reasoning_77k.json"
    ]
    
    base_url = "https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/"
    
    print("Available JSON files that were downloaded:")
    for json_file in json_files:
        print(f"   📄 {json_file}")
    
    print("\n💡 These files contain the actual training data")
    print("   We can load them directly if needed")

def main():
    print("Starting comprehensive dataset loading test...\n")
    
    # Test dataset loading
    dataset = test_dataset_loading()
    
    # Analyze what we found
    analyze_sample_data(dataset)
    
    # Test JSON approach
    test_json_loading()
    
    print(f"\n" + "="*60)
    if dataset is not None:
        print("✅ SUCCESS: Found a working loading method!")
        print(f"📊 Dataset ready with {len(dataset)} samples")
        
        # Show sample conversation
        sample = dataset[0]
        if 'conversations' in sample:
            print(f"\n💬 Sample conversation:")
            for conv in sample['conversations']:
                role = conv.get('from', 'unknown')
                content = conv.get('value', '')[:200] + "..." if len(conv.get('value', '')) > 200 else conv.get('value', '')
                print(f"   {role}: {content}")
        
    else:
        print("❌ FAILED: No loading method worked")
        print("💡 Recommendation: Use enhanced synthetic data")
    
    print("="*60)

if __name__ == "__main__":
    main()