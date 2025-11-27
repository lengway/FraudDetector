"""Quick GPU test for CatBoost"""
import sys

print("=" * 60)
print("GPU DETECTION TEST")
print("=" * 60)

# Test 1: Check NVIDIA GPU
print("\n1. Checking for NVIDIA GPU...")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0 and result.stdout.strip():
        print(f"   ✅ NVIDIA GPU found: {result.stdout.strip()}")
    else:
        print("   ⚠️ nvidia-smi failed or no GPU")
except Exception as e:
    print(f"   ⚠️ nvidia-smi not available: {e}")

# Test 2: Check CatBoost GPU support
print("\n2. Testing CatBoost GPU support...")
try:
    from catboost import CatBoostClassifier
    import numpy as np
    
    print("   Creating test model with GPU...")
    model = CatBoostClassifier(
        iterations=1,
        task_type='GPU',
        devices='0',
        verbose=False
    )
    
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    
    print("   Training on GPU...")
    model.fit(X, y, verbose=False)
    
    print("   ✅ GPU WORKS! CatBoost can use GPU")
    
except Exception as e:
    print(f"   ❌ GPU failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    
    # Fallback test with CPU
    print("\n3. Testing fallback to CPU...")
    try:
        model_cpu = CatBoostClassifier(
            iterations=1,
            task_type='CPU',
            verbose=False
        )
        model_cpu.fit(X, y, verbose=False)
        print("   ✅ CPU works (fallback successful)")
    except Exception as e2:
        print(f"   ❌ CPU also failed: {e2}")

# Test 3: Check CUDA availability (if torch installed)
print("\n4. Checking PyTorch CUDA (optional)...")
try:
    import torch
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA devices: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("   ⚠️ PyTorch not installed (not required)")
except Exception as e:
    print(f"   ⚠️ PyTorch CUDA check failed: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

import src.config as config
if config.USE_GPU:
    print("config.USE_GPU = True")
    print("  → Will attempt GPU, fallback to CPU if unavailable")
else:
    print("config.USE_GPU = False")
    print("  → Using CPU only (change to True to try GPU)")

print("=" * 60)
