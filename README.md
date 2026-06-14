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

## Build Package

**Requirements**

* avabm Conda environment
* Python 3.12
* CUDA
* NVIDIA GPU and drivers
* PyTorch
* Visual Studio 2022 with C++ build tools
* Windows environment

Run `avabm_cuda/build.bat`. A compiled `.pyd` file will be generated in the CUDA directory, and the script will automatically copy it to the project root. The script removes stale `.pyd` files before building and stops on compile errors instead of copying an old binary.


## Current Fix Notes (v22 / route cache v33)

This package includes source-level fixes for the remaining node-transition and missed-exit deadlocks, plus a v22 CUDA build hotfix for the missing `dt` argument in the priority-gate candidate kernel launch. Rebuild the CUDA extension after extracting the project:

```
avabm_cuda\build.bat
```

The route cache version was bumped, so old ready-route caches are not reused. Vehicles that miss a required left/right exit now continue through the best straight continuation instead of waiting indefinitely or cutting across lanes, and through traffic on 3+ lane roads spreads away from the right-edge ramp lane when it is not preparing for a nearby exit. `run.bat` now stops if the compiled `.pyd` is missing or older than the CUDA source so stale binaries are not used by accident.

## Run

**Requirements**

* avabm Conda environment
* avabm\_cuda Python package (compiled binary)

Execute `run.bat` to start the simulation. A Pygame window will open.