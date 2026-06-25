# AVABM

GPU/CPU Autonomous Vehicle Agent-Based Model.

This build exposes one Python package, `avabm`, with two selectable runtime
backends built as internal extension modules:

- `cuda`: the original CUDA/PyTorch extension backend.
- `cpu`: a C++ `std::thread` parallel backend that reuses the CUDA ECS core
  kernels through a source-level CPU compatibility layer for CPU fallback and
  CPU-vs-GPU analysis runs.

External code should import `avabm`, not separate `avabm_cpu` or `avabm_cuda`
packages. The legacy engine names `exact`, `micro`, `abm`, `vehicle`, and
`cuda` remain compatibility aliases. Runtime backend selection is controlled by
`SIM_BACKEND`, not by a separate non-ECS engine path.

## Environment

### Common requirements

- Python 3.12+
- PyTorch
- Python packages used by the simulator, for example:

```bash
python -m pip install numpy torch pygame matplotlib pandas scipy fastapi uvicorn PyOpenGL geopandas networkx shapely pyproj
```

For CUDA runs, also install an NVIDIA GPU driver, CUDA Toolkit with `nvcc`, and a
PyTorch build that supports your CUDA runtime. For the optional renderer packages
mentioned by older builds:

```bash
python -m pip install wgpu rendercanvas glfw
```

### Windows-only CUDA build requirement

Install Visual Studio 2022 Build Tools with the Desktop C++ workload. The default
`config.txt` uses `MSVC_TOOLSET=v143` to avoid unsupported MSVC toolsets for CUDA.
The CPU backend also needs a working C++ compiler when it is built from source.

## Backend selection

Configure the backend in `config.txt`:

```text
SIM_BACKEND=auto
CPU_WORKERS=0
CPU_AUTO_BUILD=1
```

`SIM_BACKEND` accepts:

- `auto`: try CUDA first when `torch.cuda.is_available()` is true; otherwise use
  the C++ CPU backend. If the CUDA extension is unavailable or fails to import,
  runtime falls back to CPU.
- `cuda`: prefer CUDA, but fall back to CPU when CUDA is unavailable.
- `cuda_strict`: require CUDA and fail instead of falling back.
- `cpu`: force the C++ CPU backend.

`CPU_WORKERS=0` uses hardware concurrency. Set it to a positive integer, such as
`CPU_WORKERS=8`, to cap the CPU backend worker threads. The same value is also
applied to `torch.set_num_threads()` when the CPU backend is selected.

Command-line aliases are also available:

```bash
./run.sh turbo --cpu
./run.sh turbo --backend=cpu
./run.sh turbo --backend=cuda
```

```bat
run.bat turbo --cpu
run.bat turbo --backend=cpu
run.bat turbo --backend=cuda
```

## Windows: build and run

Use the single launcher:

```bat
run.bat
```

The menu lets you choose:

1. Turbo, a headless selected-backend batch run.
2. Visual, an OpenGL/Pygame selected-backend window run.
3. Build the full `avabm` package, meaning CPU and CUDA extensions.
4. Build only the backend selected by `SIM_BACKEND`.
5. Build CUDA only, skipping when the extension is already up to date.
6. Build CPU only.
7. Clean rebuild CUDA.
8. Hard clean plus rebuild CUDA.
9. CUDA build status.

Direct commands are also supported:

```bat
run.bat turbo
run.bat visual
run.bat build
run.bat build-selected
run.bat build-cuda
run.bat build-cpu
run.bat rebuild
run.bat hardclean
run.bat status
```

The launcher checks the selected extension before running and builds automatically
when the compiled module is missing or stale.

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
./run.sh build-selected
./run.sh build-cuda
./run.sh build-cpu
./run.sh rebuild
./run.sh hardclean
./run.sh status
```

`run.sh` activates the `CONDA_ENV` from `config.txt` when conda is available, but
it can also run with the current shell Python. CUDA extension outputs are placed
inside the single `avabm/` package directory on both Linux and Windows.

## Manual CPU backend build

The CPU extension is a source-level C++ port of the CUDA ECS core. It does not include PyTorch C++ headers, so it
builds quickly with the standard Python extension toolchain:

```bash
cd avabm/cpu
python setup.py build_ext --inplace
```

The extension is installed as `avabm.avabm_cpu_ext` inside the unified package.
It exports the same simulation calls used by `main.py`: `step`, `step_batch`,
`set_num_threads`, `get_num_threads`, and CUDA-compatible render no-op/count
helpers.

## Configuration

Most build and runtime options live in `config.txt`. Useful runtime switches:

- `SIM_BACKEND=auto|cuda|cuda_strict|cpu` selects the simulation backend.
- `CPU_WORKERS=<N>` sets CPU backend worker threads; `0` means hardware
  concurrency.
- `HEADLESS_MODE=1` or `./run.sh turbo` / `run.bat turbo` for headless turbo mode.
- `HEADLESS_MODE=0` or `./run.sh visual` / `run.bat visual` for visual mode.
- `SIM_ENGINE=ecs_abm_cuda` remains the canonical engine compatibility setting;
  `SIM_BACKEND` controls CUDA versus CPU.

## Importing from Python

```python
import avabm

sim, backend = avabm.select_backend("auto")
sim.step(...)  # the selected CPU or CUDA extension API
```

For application code, use `import avabm`. The CPU and CUDA binaries remain
separate compiled extension modules internally, but they live under the same
package namespace.

## CUDA build stability notes

This package disables runtime JIT compilation for CUDA. The launchers build
explicitly and stamp a source/config fingerprint. Normal repeated CUDA builds are
skipped once the compiled extension matches the current source and build config.

The default CUDA profile keeps ptxas inputs partitioned for stability:

```text
CUDA_OPT_LEVEL=2
CUDA_PTXAS_OPT_LEVEL=1
CUDA_PTXAS_SAFE_MODE=0
CUDA_PTXAS_PARTITION_BUILD=1
CUDA_SINGLE_TU_BUILD=0
```

## Benchmark mode

Use benchmark mode to run the same scenario once on the C++ CPU backend and once
on the CUDA backend, then print and save a throughput comparison.

```bash
./run.sh benchmark
```

```bat
run.bat benchmark
```

You can also run one side only:

```bash
./run.sh benchmark cpu
./run.sh benchmark gpu
./run.sh benchmark --benchmark-order=cuda
```

```bat
run.bat benchmark cpu
run.bat benchmark gpu
run.bat benchmark --benchmark-order=cuda
```

Defaults are configured in `config.txt`:

```text
BENCHMARK_STEPS=10000
BENCHMARK_ORDER=cpu,cuda
BENCHMARK_FIXED_SPAWN=1
BENCHMARK_WARMUP_STEPS=1000
BENCHMARK_BATCH_STEPS=16384
BENCHMARK_CPU_WORKERS=0
BENCHMARK_OUTPUT_DIR=data/results
BENCHMARK_SAVE_CHILD_METRICS=0
```

`BENCHMARK_FIXED_SPAWN=1` disables SPWNxx time-profile demand during the
benchmark run, so both child runs use the same constant default spawn demand
computed from `BASE_VPS`, lane/road-width multipliers, and `MAX_TOTAL_VPS`.
Each child process also receives the same deterministic benchmark RNG state from
`SCENARIO_SEED`. `BENCHMARK_WARMUP_STEPS` runs before timing and resets metrics,
which fills the road and reduces cold-start GPU utilization dips.
`BENCHMARK_BATCH_STEPS` is intentionally large so CUDA can queue a long run with
fewer host-side gaps. `BENCHMARK_CPU_WORKERS=0` resolves to all logical CPU
threads and is passed to both Torch and the C++ CPU backend.

Results are written to:

- `data/results/benchmark_summary.json`
- `data/results/benchmark_summary.csv`
- `data/results/benchmark_01_cpu.log`
- `data/results/benchmark_02_cuda.log`

To override the step count without editing the config:

```bash
./run.sh benchmark --benchmark-steps=20000
```

```bat
run.bat benchmark --benchmark-steps=20000
```
