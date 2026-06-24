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

The build scripts now default to a conservative ptxas-safe profile for Windows systems where large CUDA builds can crash with `ptxas` `ACCESS_VIOLATION`, especially on sm_86/Ampere targets. The old monolithic CUDA `main.cu` has also been split into smaller CUDA translation units (`*.cu`) so each nvcc/ptxas invocation handles less device code. The build fingerprint includes the resolved GPU architecture, CUDA Toolkit path, nvcc version, and all split CUDA sources/headers, so binaries built for one GPU/toolkit are not silently reused on another. See `AVABM_BUILD_STABILITY_NOTES_KO.md` and `avabm_cuda/CUDA_SPLIT_MAP.md` for details. The split CUDA files use short names such as `common.cuh`, `priority.cu`, and `render.cu`; the Python extension remains named `avabm_cuda`.


**Requirements**

* avabm Conda environment
* Python 3.12
* CUDA
* NVIDIA GPU and drivers
* PyTorch
* Visual Studio 2022 with C++ build tools
* Windows environment

Run `avabm_cuda/build.bat`. A compiled `.pyd` file will be generated in the CUDA directory, and the script will automatically copy it to the project root. The script removes stale `.pyd` files before building and stops on compile errors instead of copying an old binary.


## Current Fix Notes (v26 / route cache v40)

This package includes source-level fixes for node-transition gridlock, connector-exit stalls, and bumper-overlap recovery. Rebuild the CUDA extension after extracting the project:

```
avabm_cuda\build.bat
```

The route cache version was bumped, so old ready-route caches are not reused. The v26 CUDA traffic logic adds a deterministic node-drain rule: when conflicting vehicles have queued at the same node, a clear-front vehicle that has waited longer is released instead of allowing an A-waits-B / B-waits-C cycle to persist. Connector vehicles that are already at the exit can perform a tiny-gap handoff so they do not occupy the intersection box forever. Contact repair was strengthened with larger longitudinal separation, five repair passes, and bounded front-vehicle nudges when a rear vehicle is clamped at the lane or connector origin. `run.bat` still stops if the compiled `.pyd` is missing or older than the CUDA source so stale binaries are not used by accident.

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

If RTX A4000/sm_86 ptxas crashes again, see `AVABM_CUDA_PERFORMANCE_TUNING_KO.md` and temporarily switch back to the ultra-safe profile.
