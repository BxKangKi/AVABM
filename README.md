# AVABM

GPU-based Autonomous Vehicle Agent-Based Model.

This build uses one ECS ABM CUDA simulation engine. The legacy names `exact`,
`micro`, `abm`, `vehicle`, and `cuda` are compatibility aliases that all select
`ecs_abm_cuda`; there is no separate non-ECS `exact` engine path in this tree.

## Environment

### Common requirements

- NVIDIA GPU and driver
- CUDA Toolkit with `nvcc`
- Python 3.12
- PyTorch built for your CUDA runtime
- Python packages used by the simulator, for example:

```bash
python -m pip install numpy torch pygame pybind11 ninja matplotlib pandas scipy fastapi uvicorn PyOpenGL geopandas networkx shapely pyproj
```

For the optional renderer packages mentioned by older builds:

```bash
python -m pip install wgpu rendercanvas glfw
```

### Windows-only build requirement

Install Visual Studio 2022 Build Tools with the Desktop C++ workload. The default
`config.txt` uses `MSVC_TOOLSET=v143` to avoid unsupported MSVC toolsets for CUDA.

## Windows: build and run

Use the single launcher:

```bat
run.bat
```

The menu lets you choose:

1. Turbo, a headless CUDA batch run.
2. Visual, an OpenGL/Pygame window run.
3. Build CUDA only, skipping when the extension is already up to date.
4. Clean rebuild CUDA.
5. Hard clean plus rebuild CUDA.
6. CUDA build status.

Direct commands are also supported:

```bat
run.bat turbo
run.bat visual
run.bat build
run.bat rebuild
run.bat hardclean
run.bat status
```

The old helper batch files were consolidated into `run.bat`. The launcher checks
the CUDA extension fingerprint before running and builds automatically if the
compiled module is missing or stale.

## Linux: build and run

Make the launcher executable once:

```bash
chmod +x run.sh
```

Then use the same style of launcher commands:

```bash
./run.sh
./run.sh turbo
./run.sh visual
./run.sh build
./run.sh rebuild
./run.sh hardclean
./run.sh status
```

`run.sh` activates the `CONDA_ENV` from `config.txt` when conda is available, but
it can also run with the current shell Python. It supports Linux CUDA extension
outputs (`avabm_cuda*.so`) as well as Windows outputs (`avabm_cuda*.pyd`).

## Configuration

Most build and runtime options live in `config.txt`. Useful runtime switches:

- `HEADLESS_MODE=1` or `./run.sh turbo` / `run.bat turbo` for headless turbo mode.
- `HEADLESS_MODE=0` or `./run.sh visual` / `run.bat visual` for visual mode.
- `SIM_ENGINE=ecs_abm_cuda` is the canonical engine setting.

## CUDA build stability notes

This package disables runtime JIT compilation. The launchers build explicitly and
stamp a source/config fingerprint. Normal repeated builds are skipped once the
compiled extension matches the current source and build config.

The default CUDA profile keeps ptxas inputs partitioned for stability:

```text
CUDA_OPT_LEVEL=2
CUDA_PTXAS_OPT_LEVEL=1
CUDA_PTXAS_SAFE_MODE=0
CUDA_PTXAS_PARTITION_BUILD=1
CUDA_SINGLE_TU_BUILD=0
```
