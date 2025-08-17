#!/usr/bin/env python3
"""
Training Monitor
Checks if training is still running and healthy
"""

import subprocess
import time
import torch

print("📊 LLaVA Training Monitor")
print("=" * 30)

def check_gpu_status():
    """Check GPU memory and processes"""
    try:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"🖥️ GPU Memory:")
            print(f"  - Allocated: {allocated:.1f}GB")
            print(f"  - Reserved: {reserved:.1f}GB") 
            print(f"  - Total: {total:.1f}GB")
            print(f"  - Utilization: {(allocated/total)*100:.1f}%")
            
            if allocated < 1.0:
                print("⚠️ Very low GPU memory - training likely crashed")
                return False
            elif allocated > 1.0:
                print("✅ GPU memory looks healthy")
                return True
        else:
            print("❌ No CUDA GPU available")
            return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def check_processes():
    """Check for training processes"""
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        processes = result.stdout
        
        python_processes = [line for line in processes.split('\n') 
                          if 'python' in line and 'train_llava' in line]
        
        print(f"\n🔍 Training Processes:")
        if python_processes:
            for proc in python_processes:
                print(f"  ✅ {proc.strip()}")
            return True
        else:
            print("  ❌ No training processes found")
            return False
            
    except Exception as e:
        print(f"❌ Error checking processes: {e}")
        return False

def check_nvidia_smi():
    """Check nvidia-smi output"""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            
            print(f"\n🖥️ NVIDIA-SMI Status:")
            
            # Look for python processes
            python_found = False
            for line in lines:
                if 'python' in line.lower():
                    print(f"  ✅ {line.strip()}")
                    python_found = True
                    
            if not python_found:
                print("  ❌ No Python processes using GPU")
                
            return python_found
        else:
            print("❌ nvidia-smi failed")
            return False
            
    except Exception as e:
        print(f"❌ Error running nvidia-smi: {e}")
        return False

# Run all checks
print("Running health checks...\n")

gpu_healthy = check_gpu_status()
processes_running = check_processes() 
gpu_processes = check_nvidia_smi()

print(f"\n" + "=" * 30)
print("📋 Summary:")
print(f"  GPU Memory: {'✅ Healthy' if gpu_healthy else '❌ Problem'}")
print(f"  Processes: {'✅ Running' if processes_running else '❌ Not found'}")
print(f"  GPU Usage: {'✅ Active' if gpu_processes else '❌ Inactive'}")

if gpu_healthy and processes_running and gpu_processes:
    print("\n🎉 Training appears to be running normally!")
elif not processes_running:
    print("\n💀 Training process appears to have crashed!")
    print("🔧 Suggested actions:")
    print("  1. Check the terminal output for errors")
    print("  2. Run: python resume_training.py")
    print("  3. Or restart: python train_llava_150k_ultimate.py")
else:
    print("\n⚠️ Training status unclear - manual investigation needed")

print("\n📊 To continue monitoring, run this script again in a few minutes")