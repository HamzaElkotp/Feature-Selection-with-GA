"""Configuration settings for the GA application."""
import os

# GPU Configuration
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))

# Try to import CuPy and check GPU availability
GPU_AVAILABLE = False
cp = None
try:
    import cupy as cp
    # Check if CUDA is available
    if hasattr(cp, 'cuda') and cp.cuda.is_available():
        GPU_AVAILABLE = True
        # Set the device
        try:
            cp.cuda.Device(GPU_DEVICE_ID).use()
            gpu_name = cp.cuda.runtime.getDeviceProperties(GPU_DEVICE_ID)['name'].decode('utf-8')
            print(f"GPU acceleration enabled: {gpu_name}")
        except Exception as e:
            print(f"Warning: Could not set GPU device {GPU_DEVICE_ID}: {e}")
            GPU_AVAILABLE = False
    else:
        print("GPU acceleration requested but CUDA is not available. Falling back to CPU.")
except ImportError:
    if USE_GPU:
        print("CuPy not installed. GPU acceleration disabled. Install with: pip install cupy-cuda11x")
except Exception as e:
    if USE_GPU:
        error_msg = str(e)
        if "nvrtc" in error_msg or "dll" in error_msg.lower() or "Could not find module" in error_msg:
            print("\n" + "=" * 70)
            print("CUDA Runtime Library Error Detected")
            print("=" * 70)
            print("CuPy cannot find CUDA runtime libraries (nvrtc64_112_0.dll).")
            print("\nQUICK FIX:")
            print("1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
            print("2. Restart your computer after installation")
            print("3. Reinstall CuPy: pip uninstall cupy-cuda11x && pip install cupy-cuda11x")
            print("\nFor detailed troubleshooting, see: GPU_TROUBLESHOOTING.md")
            print("\nFalling back to CPU processing (program will still work)...")
            print("=" * 70 + "\n")
        else:
            print(f"Error initializing GPU: {e}. Falling back to CPU.")
    GPU_AVAILABLE = False
    cp = None  # Ensure cp is None if initialization fails

