# AVABM

GPU-based Autonomous Vehicle Agent-Based Model

This project is currently under development.

## Generate Conda Environment

1. Install Miniforge from: <https://github.com/conda-forge/miniforge> (no administrator privileges required).
2. Create an environment using:
   ```
   conda create -n avabm python=3.12
   ```
3. Install the required packages:
   ```
   python -m pip install numpy torch pygame pybind11 ninja matplotlib pandas scipy fastapi uvicorn PyOpenGL
   ```
   (torch package should be matched by installed CUDA version)

   For the default Vulkan/WebGPU renderer in this patched build, also install:
   ```
   python -m pip install wgpu rendercanvas glfw
   ```

## Build Package

## CUDA build stability notes

This package disables runtime JIT compilation. Build the native extension explicitly with:

```bat
avabm_cuda\build.bat clean
```

When moving between GPUs or CUDA/driver installs, use a harder cache reset once:

```bat
avabm_cuda\build.bat hardclean
```

`hardclean` removes the project build outputs plus conservative user-local `torch_extensions` and NVIDIA `ComputeCache` folders before rebuilding. It does not enable runtime JIT builds.


**Requirements**

* avabm Conda environment
* Python 3.12
* CUDA
* NVIDIA GPU and drivers
* PyTorch
* Visual Studio 2022 with C++ build tools
* Windows environment

Run `avabm_cuda/build.bat`. A compiled `.pyd` file will be generated in the CUDA directory, and the script will automatically copy it to the project root. The script removes stale `.pyd` files before building and stops on compile errors instead of copying an old binary.


## Run

**Requirements**

* avabm Conda environment
* avabm\_cuda Python package (compiled binary)

Execute `run.bat` to start the simulation. A Pygame window will open.
## CUDA sm86 performance-safe profile

This package uses a performance-safe CUDA profile after splitting the former large motion translation unit:

```text
CUDA_OPT_LEVEL=2
CUDA_PTXAS_OPT_LEVEL=1
CUDA_PTXAS_SAFE_MODE=0
CUDA_DEVICE_NOINLINE=0
CUDA_FAST_MATH=0
```