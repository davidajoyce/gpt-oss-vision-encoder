#!/usr/bin/env python3
"""
Disk Space Cleanup Script
Safely frees up space on RunPod for continued training
"""

import os
import glob
import shutil
import subprocess

print("🧹 RunPod Disk Space Cleanup")
print("=" * 40)

def get_disk_usage():
    """Get current disk usage statistics"""
    try:
        statvfs = os.statvfs('.')
        total_gb = statvfs.f_frsize * statvfs.f_blocks / 1e9
        free_gb = statvfs.f_frsize * statvfs.f_bavail / 1e9
        used_gb = total_gb - free_gb
        usage_percent = (used_gb / total_gb) * 100
        
        return total_gb, used_gb, free_gb, usage_percent
    except Exception as e:
        print(f"❌ Error getting disk usage: {e}")
        return None, None, None, None

def cleanup_old_checkpoints(keep_latest=2, dry_run=False):
    """Remove old checkpoint files, keeping only the latest N"""
    pattern = "*checkpoint_step_*.pt"
    checkpoints = sorted(glob.glob(pattern))
    
    if len(checkpoints) <= keep_latest:
        print(f"✅ Only {len(checkpoints)} checkpoints found, keeping all")
        return 0
    
    to_remove = checkpoints[:-keep_latest]
    total_size = 0
    
    print(f"🗑️ Found {len(checkpoints)} checkpoints, removing {len(to_remove)} old ones:")
    
    for checkpoint in to_remove:
        try:
            size = os.path.getsize(checkpoint) / 1e9  # GB
            total_size += size
            
            if dry_run:
                print(f"  Would remove: {checkpoint} ({size:.1f}GB)")
            else:
                os.remove(checkpoint)
                print(f"  ✅ Removed: {checkpoint} ({size:.1f}GB)")
                
        except Exception as e:
            print(f"  ❌ Failed to remove {checkpoint}: {e}")
    
    return total_size

def cleanup_cache_dirs(dry_run=False):
    """Clean up various cache directories"""
    cache_dirs = [
        ("HuggingFace datasets", "~/.cache/huggingface/datasets"),
        ("HuggingFace models", "~/.cache/huggingface/transformers"),
        ("Pip cache", "~/.cache/pip"),
        ("Torch hub", "~/.cache/torch/hub"),
    ]
    
    total_size = 0
    
    for name, path in cache_dirs:
        expanded_path = os.path.expanduser(path)
        if os.path.exists(expanded_path):
            try:
                # Get directory size
                result = subprocess.run(['du', '-sh', expanded_path], 
                                      capture_output=True, text=True)
                size_str = result.stdout.split()[0] if result.returncode == 0 else "unknown"
                
                if dry_run:
                    print(f"  Would clean: {name} ({size_str}) - {expanded_path}")
                else:
                    shutil.rmtree(expanded_path)
                    print(f"  ✅ Cleaned: {name} ({size_str})")
                    
            except Exception as e:
                print(f"  ❌ Failed to clean {name}: {e}")
        else:
            print(f"  ⚪ {name} - not found")
    
    return total_size

def cleanup_temp_files(dry_run=False):
    """Clean up temporary files"""
    temp_patterns = [
        "/tmp/tmp*",
        "/tmp/pytorch_*",
        "*.tmp",
        "*.temp",
        "core.*"
    ]
    
    total_size = 0
    
    print(f"🧹 Cleaning temporary files:")
    
    for pattern in temp_patterns:
        files = glob.glob(pattern)
        for file_path in files:
            try:
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path) / 1e9
                    total_size += size
                    
                    if dry_run:
                        print(f"  Would remove: {file_path} ({size:.1f}GB)")
                    else:
                        os.remove(file_path)
                        print(f"  ✅ Removed: {file_path}")
                        
            except Exception as e:
                print(f"  ❌ Failed to remove {file_path}: {e}")
    
    return total_size

def main():
    # Check current disk usage
    total_gb, used_gb, free_gb, usage_percent = get_disk_usage()
    
    if total_gb:
        print(f"💾 Current Disk Usage:")
        print(f"  Total: {total_gb:.1f}GB")
        print(f"  Used: {used_gb:.1f}GB ({usage_percent:.1f}%)")
        print(f"  Free: {free_gb:.1f}GB")
        
        if free_gb < 5:
            print("🚨 CRITICAL: Less than 5GB free space!")
        elif free_gb < 10:
            print("⚠️ WARNING: Less than 10GB free space")
        else:
            print("✅ Disk space looks healthy")
    
    print("\n" + "=" * 40)
    
    # Ask user what to clean
    print("🧹 Cleanup Options:")
    print("1. Old checkpoints (recommended)")
    print("2. Cache directories (datasets, pip, etc.)")
    print("3. Temporary files")
    print("4. All of the above")
    print("5. Dry run (show what would be deleted)")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup cancelled by user")
        return
    
    dry_run = choice == "5"
    if dry_run:
        print("\n🔍 DRY RUN - showing what would be deleted:\n")
    else:
        print(f"\n🧹 Starting cleanup...\n")
    
    freed_space = 0
    
    if choice in ["1", "4", "5"]:
        print("🗑️ Cleaning old checkpoints:")
        freed_space += cleanup_old_checkpoints(keep_latest=2, dry_run=dry_run)
    
    if choice in ["2", "4", "5"]:
        print("\n🗂️ Cleaning cache directories:")
        freed_space += cleanup_cache_dirs(dry_run=dry_run)
    
    if choice in ["3", "4", "5"]:
        print("\n🧹 Cleaning temporary files:")
        freed_space += cleanup_temp_files(dry_run=dry_run)
    
    print("\n" + "=" * 40)
    
    if dry_run:
        print("🔍 DRY RUN COMPLETE - no files were actually deleted")
    else:
        print(f"✅ Cleanup complete!")
        
        # Check disk usage again
        new_total_gb, new_used_gb, new_free_gb, new_usage_percent = get_disk_usage()
        if new_free_gb and free_gb:
            space_freed = new_free_gb - free_gb
            print(f"💾 Space freed: {space_freed:.1f}GB")
            print(f"💾 New free space: {new_free_gb:.1f}GB")
            
            if new_free_gb > 10:
                print("🎉 Disk space is now healthy for training!")
            elif new_free_gb > 5:
                print("✅ Should have enough space to continue training")
            else:
                print("⚠️ Still low on space - consider increasing pod storage")
    
    print("\n🚀 Ready to resume training:")
    print("python monitor_training.py    # Check if training is still running")
    print("python resume_training.py     # Resume from latest checkpoint")

if __name__ == "__main__":
    main()