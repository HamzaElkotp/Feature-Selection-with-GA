# Feature Extraction with Genetic Algorithm

A genetic algorithm-based feature selection tool with GPU acceleration support.

## Features

- Genetic Algorithm for feature selection
- Multiple selection methods (Tournament, Roulette Wheel, Random)
- Multiple mutation methods (Bit Flip, Complement, Reverse, Rotation)
- GPU acceleration support for faster processing (NVIDIA GPUs)
- GUI interface for easy interaction

## GPU Acceleration

This project supports GPU acceleration using CuPy for NVIDIA GPUs (CUDA-compatible). GPU acceleration can significantly speed up:
- Population fitness computation
- Crossover operations
- Mutation operations
- Array-based operations

### Prerequisites for GPU

- NVIDIA GPU with CUDA support (e.g., RTX 3050)
- CUDA Toolkit 11.x or 12.x installed
- cuDNN (optional, but recommended)

### Installing GPU Support

1. **Install CUDA Toolkit** (if not already installed):
   - Download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
   - For RTX 3050, CUDA 11.x or 12.x is recommended

2. **Install CuPy**:
   ```bash
   # For CUDA 11.x (recommended for RTX 3050)
   pip install cupy-cuda11x
   
   # OR for CUDA 12.x
   pip install cupy-cuda12x
   ```

3. **Verify GPU Installation**:
   ```python
   import cupy as cp
   print(cp.cuda.is_available())  # Should print True
   print(cp.cuda.runtime.getDeviceProperties(0)['name'])  # Should show your GPU name
   ```

### Enabling/Disabling GPU

GPU acceleration is enabled by default if a compatible GPU is detected. You can control it using environment variables:

```bash
# Disable GPU (force CPU)
export USE_GPU=false

# Enable GPU (default)
export USE_GPU=true

# Specify GPU device ID (if you have multiple GPUs)
export GPU_DEVICE_ID=0
```

Or set it in Python:
```python
import os
os.environ["USE_GPU"] = "true"  # or "false"
```

## Installation

1. **Clone the repository** (or navigate to the project directory)

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

   This will install:
   - NumPy
   - Pandas
   - scikit-learn
   - ttkbootstrap (for GUI)
   - CuPy (if CUDA is available)

3. **Run the application**:
   ```bash
   python main.py
   ```

## Usage

1. Launch the application using `python main.py`
2. Select your dataset file (CSV format)
3. Configure GA parameters:
   - Population size
   - Number of generations
   - Selection method
   - Mutation method
   - Crossover points
   - Alpha and Beta values
4. Run the genetic algorithm
5. View results and best feature combinations

## Performance Notes

- **GPU Acceleration**: GPU acceleration provides the most benefit for:
  - Large populations (>50 individuals)
  - Many generations (>50)
  - Large feature sets (>100 features)
  
- **CPU Fallback**: The system automatically falls back to CPU if:
  - GPU is not available
  - CuPy is not installed
  - GPU operations fail

- **Hybrid Processing**: Some operations (like ML model training) still run on CPU, but array operations are GPU-accelerated for better performance.

## Troubleshooting

### Error: "CuPy failed to load nvrtc64_112_0.dll"

This means CUDA Toolkit is not installed or not in PATH. **Quick fix:**

1. **Install CUDA Toolkit**:
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Choose CUDA 11.8 or 12.x for RTX 3050
   - Run installer and choose "Express Installation"
   - **Restart your computer** after installation

2. **Verify installation**:
   ```bash
   nvcc --version
   ```

3. **Reinstall CuPy**:
   ```bash
   pip uninstall cupy-cuda11x
   pip install cupy-cuda11x
   ```

4. **Temporary workaround** (use CPU mode):
   ```bash
   set USE_GPU=false
   ```

**For detailed troubleshooting, see `GPU_TROUBLESHOOTING.md`**

### GPU not detected
- Ensure CUDA Toolkit is installed and in PATH
- Verify GPU drivers are up to date
- Check that `cupy-cuda11x` or `cupy-cuda12x` matches your CUDA version
- Restart computer after installing CUDA

### Out of Memory errors
- Reduce population size
- Process in smaller batches
- Close other GPU-intensive applications

### Performance not improved
- GPU acceleration is most beneficial for large populations and many features
- Check that GPU is actually being used (check GPU utilization with `nvidia-smi`)
- Some operations (ML model training) still use CPU

## License

See LICENSE file for details.

