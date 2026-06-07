from pathlib import Path
import os
import shlex
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _read_config(path: Path):
    cfg = {}
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].split(";", 1)[0].strip().strip('"').strip("'")
        if key:
            cfg[key] = None if value.lower() in {"", "none", "null"} else os.path.expandvars(value)
    return cfg


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG = _read_config(ROOT_DIR / "config.txt")

if CONFIG.get("CUDA_HOME") and not os.environ.get("CUDA_HOME"):
    os.environ["CUDA_HOME"] = CONFIG["CUDA_HOME"]
if CONFIG.get("MAX_JOBS") and not os.environ.get("MAX_JOBS"):
    os.environ["MAX_JOBS"] = CONFIG["MAX_JOBS"]
if CONFIG.get("TORCH_NVCC_FLAGS") and not os.environ.get("TORCH_NVCC_FLAGS"):
    os.environ["TORCH_NVCC_FLAGS"] = CONFIG["TORCH_NVCC_FLAGS"]

nvcc_flags = ["-O3", "--use_fast_math"]
extra_nvcc = CONFIG.get("CUDA_EXTRA_NVCC_FLAGS")
if extra_nvcc:
    nvcc_flags.extend(shlex.split(extra_nvcc))

cxx_flags = ["-O3"]
extra_cxx = CONFIG.get("CUDA_EXTRA_CXX_FLAGS")
if extra_cxx:
    cxx_flags.extend(shlex.split(extra_cxx))

# EN: Build with: python setup.py build_ext --inplace
# KO: 빌드: python setup.py build_ext --inplace
setup(
    name="avabm_cuda",
    ext_modules=[
        CUDAExtension(
            name="avabm_cuda",
            sources=["binding.cpp", "main.cu"],
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
