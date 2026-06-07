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

Run `avabm_cuda/build.bat`. A compiled `.pyd` file will be generated in the same directory, and the script will automatically copy it to the parent directory.

## Run

**Requirements**

* avabm Conda environment
* avabm\_cuda Python package (compiled binary)

Execute `run.bat` to start the simulation. A Pygame window will open.