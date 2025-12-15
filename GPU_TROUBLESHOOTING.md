# GPU Troubleshooting Guide

## Error: "CuPy failed to load nvrtc64_112_0.dll"

This error means CuPy cannot find the CUDA runtime libraries. Here's how to fix it:

### Solution 1: Install CUDA Toolkit

1. **Download CUDA Toolkit**:
   - Go to: https://developer.nvidia.com/cuda-downloads
   - Select your OS (Windows), architecture (x86_64), version (Windows 10/11), and installer type (exe local)
   - Download CUDA Toolkit 11.8 or 12.x (recommended for RTX 3050)

2. **Install CUDA Toolkit**:
   - Run the installer
   - Choose "Express Installation" (recommended)
   - The installer will add CUDA to your system PATH automatically

3. **Verify Installation**:
   ```bash
   nvcc --version
   ```
   This should show your CUDA version.

4. **Restart your computer** after installation

5. **Reinstall CuPy**:
   ```bash
   pip uninstall cupy-cuda11x
   pip install cupy-cuda11x
   ```

### Solution 2: Add CUDA to PATH (if already installed)

If CUDA is installed but not in PATH:

1. Find your CUDA installation (usually `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin`)
2. Add it to Windows PATH:
   - Press `Win + R`, type `sysdm.cpl`, press Enter
   - Go to "Advanced" tab → "Environment Variables"
   - Under "System variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.x\bin`
   - Replace `v11.x` with your actual CUDA version
   - Click OK on all dialogs
   - **Restart your computer**

### Solution 3: Use CPU Mode (Temporary Workaround)

If you want to run the program without GPU for now:

1. **Set environment variable**:
   ```bash
   set USE_GPU=false
   ```
   Or in PowerShell:
   ```powershell
   $env:USE_GPU="false"
   ```

2. **Or modify the code** to disable GPU by default:
   - Edit `config/settings.py`
   - Change: `USE_GPU = os.getenv("USE_GPU", "false").lower() == "true"`

### Solution 4: Check CUDA Version Compatibility

Make sure your CuPy version matches your CUDA version:

- **CUDA 11.x** → Use `cupy-cuda11x`
- **CUDA 12.x** → Use `cupy-cuda12x`

Check your CUDA version:
```bash
nvcc --version
```

Then install the matching CuPy:
```bash
# For CUDA 11.x
pip uninstall cupy
pip install cupy-cuda11x

# For CUDA 12.x
pip uninstall cupy
pip install cupy-cuda12x
```

### Verify GPU Setup

After fixing, verify everything works:

```python
import cupy as cp
print("CuPy version:", cp.__version__)
print("CUDA available:", cp.cuda.is_available())
if cp.cuda.is_available():
    print("GPU name:", cp.cuda.runtime.getDeviceProperties(0)['name'])
```

### Common Issues

**Issue**: "CUDA is installed but still getting error"
- **Fix**: Restart your computer after installing CUDA
- **Fix**: Make sure you're using the correct Python environment

**Issue**: "Multiple CUDA versions installed"
- **Fix**: Use the PATH to point to the version you want
- **Fix**: Install the matching CuPy version

**Issue**: "GPU not detected"
- **Fix**: Make sure NVIDIA drivers are up to date
- **Fix**: Check GPU with `nvidia-smi` command

### Still Having Issues?

1. Check NVIDIA driver version: `nvidia-smi`
2. Verify CUDA installation: `nvcc --version`
3. Check Python environment: `python --version`
4. Try reinstalling CuPy: `pip uninstall cupy-cuda11x && pip install cupy-cuda11x`
5. Use CPU mode as temporary workaround (set `USE_GPU=false`)

The program will automatically fall back to CPU processing if GPU is not available, so you can still use it while fixing GPU setup.

