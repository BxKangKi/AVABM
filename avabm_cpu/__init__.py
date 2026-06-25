"""CPU backend for AVABM.

This package exposes the same step/step_batch API shape as avabm_cuda.  The
extension runs on contiguous CPU torch tensors and executes the CUDA ECS core
through a source-level C++ compatibility layer with a configurable std::thread
launcher.
"""
try:
    from .avabm_cpu_ext import *  # noqa: F401,F403
except ImportError as exc:  # pragma: no cover - gives a clearer runtime error
    raise ImportError(
        "avabm_cpu extension is not built. Run `(cd avabm_cpu && python setup.py build_ext --inplace)` "
        "or let main.py build it with CPU_AUTO_BUILD=1."
    ) from exc
