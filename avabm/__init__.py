"""Unified AVABM backend package.

Use ``import avabm`` at the application boundary.  CPU and CUDA are built as
binary extension modules inside this one package:

- ``avabm.avabm_cpu_ext``: C++ CPU backend.
- ``avabm.avabm_cuda``: CUDA/PyTorch backend.

The public helper functions below keep backend selection in one place while the
simulation API (``step``, ``step_batch``, render helpers, thread controls, ...)
is delegated to the selected backend module.
"""
from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Iterable

__version__ = "0.3.0"

_CPU_EXT = ".avabm_cpu_ext"
_CUDA_EXT = ".avabm_cuda"
_BACKEND_ALIASES = {
    "": "auto",
    "auto": "auto",
    "gpu": "cuda",
    "cuda_gpu": "cuda",
    "cuda": "cuda",
    "cuda_strict": "cuda_strict",
    "strict_cuda": "cuda_strict",
    "cpu": "cpu",
    "cpp_cpu": "cpu",
    "c++": "cpu",
}

_selected_backend_name: str | None = None
_selected_backend_module: ModuleType | None = None


def _normalize_backend_name(name: str | None) -> str:
    requested = str(name if name is not None else os.environ.get("SIM_BACKEND", "auto")).strip().lower()
    return _BACKEND_ALIASES.get(requested, "auto")


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def import_cpu() -> ModuleType:
    """Import and return the AVABM CPU extension module."""
    try:
        return importlib.import_module(_CPU_EXT, __name__)
    except ImportError as exc:
        raise ImportError(
            "AVABM CPU extension is not built. Run `run.bat build-cpu` / "
            "`./run.sh build-cpu`, or manually run "
            "`cd avabm/cpu && python setup.py build_ext --inplace`."
        ) from exc


def import_cuda() -> ModuleType:
    """Import and return the AVABM CUDA extension module."""
    try:
        return importlib.import_module(_CUDA_EXT, __name__)
    except ImportError as exc:
        raise ImportError(
            "AVABM CUDA extension is not built or its CUDA/PyTorch DLLs are not loadable. "
            "Run `run.bat build-cuda` / `./run.sh build-cuda`, or run `run.bat build` "
            "/ `./run.sh build` to build both package backends."
        ) from exc


def select_backend(name: str | None = None, *, cuda_available: bool | None = None) -> tuple[ModuleType, str]:
    """Select a backend and return ``(module, backend_name)``.

    ``name`` accepts ``auto``, ``cuda``, ``cuda_strict`` and ``cpu`` plus the
    compatibility aliases used by older config files.  ``cuda`` and ``auto``
    fall back to CPU if CUDA cannot be imported; ``cuda_strict`` raises.
    """
    global _selected_backend_name, _selected_backend_module

    requested = _normalize_backend_name(name)
    strict_cuda = requested == "cuda_strict"
    can_try_cuda = requested in {"auto", "cuda", "cuda_strict"}
    cuda_ok = _torch_cuda_available() if cuda_available is None else bool(cuda_available)

    if can_try_cuda:
        if cuda_ok:
            try:
                module = import_cuda()
                _selected_backend_name = "cuda"
                _selected_backend_module = module
                return module, "cuda"
            except ImportError:
                if strict_cuda:
                    raise
        elif strict_cuda:
            raise RuntimeError("SIM_BACKEND=cuda_strict was requested, but torch.cuda.is_available() is False.")

    module = import_cpu()
    _selected_backend_name = "cpu"
    _selected_backend_module = module
    return module, "cpu"


def get_backend() -> ModuleType:
    """Return the currently selected backend, selecting from ``SIM_BACKEND`` lazily."""
    if _selected_backend_module is None:
        select_backend(os.environ.get("SIM_BACKEND", "auto"))
    assert _selected_backend_module is not None
    return _selected_backend_module


def backend_name() -> str:
    """Return the selected backend name, selecting lazily if needed."""
    if _selected_backend_name is None:
        select_backend(os.environ.get("SIM_BACKEND", "auto"))
    assert _selected_backend_name is not None
    return _selected_backend_name


def available_backends() -> tuple[str, ...]:
    """Return backend extensions that can be imported in the current process."""
    available: list[str] = []
    for name, loader in (("cpu", import_cpu), ("cuda", import_cuda)):
        try:
            loader()
        except Exception:
            continue
        available.append(name)
    return tuple(available)


def __getattr__(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    module = get_backend()
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module 'avabm' has no attribute {name!r}") from exc


def __dir__() -> Iterable[str]:
    base = set(globals())
    try:
        base.update(dir(get_backend()))
    except Exception:
        pass
    return sorted(base)


__all__ = [
    "__version__",
    "available_backends",
    "backend_name",
    "get_backend",
    "import_cpu",
    "import_cuda",
    "select_backend",
]
