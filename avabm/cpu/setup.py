import os
import hashlib
from pathlib import Path
from setuptools import setup, Extension

SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parents[1]
PACKAGE_DIR = ROOT / "avabm"
CONFIG_PATH = ROOT / "config.txt"


def _read_config(path: Path):
    cfg = {}
    if path.exists():
        for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip().upper()] = v.strip()
    return cfg


def _cfg(cfg, name, default=None):
    return os.environ.get(name, cfg.get(name, default))


def _truthy(value, default=False):
    if value is None:
        return default
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on", "y"}:
        return True
    if s in {"0", "false", "no", "off", "n"}:
        return False
    return default


def _int_value(value, default=0):
    try:
        return int(float(str(value).strip()))
    except Exception:
        return int(default)


def _float_value(value, default=0.0):
    try:
        return float(str(value).strip())
    except Exception:
        return float(default)


cfg = _read_config(CONFIG_PATH)
std = str(_cfg(cfg, "CPU_CXX_STANDARD", _cfg(cfg, "CUDA_CXX_STANDARD", "17"))).strip() or "17"
if std not in {"17", "20"}:
    std = "17"
opt = str(_cfg(cfg, "CPU_OPT_LEVEL", "2")).strip().upper().lstrip("O")
if opt not in {"0", "1", "2", "3", "S", "G"}:
    opt = "2"
extra = str(_cfg(cfg, "CPU_EXTRA_CXX_FLAGS", "")).split()

fast_physics_mode = _truthy(_cfg(cfg, "CUDA_FAST_PHYSICS_MODE", "0"), False)
lane_hash_default = fast_physics_mode
safety_interval = max(1, _int_value(_cfg(cfg, "CUDA_SAFETY_METRICS_INTERVAL", "1"), 1))

defines = [
    ("AVABM_CPU_CUDA_SOURCE_PARITY", "1"),
    ("AVABM_CPU_PARALLEL_STATS_ENABLED", "1" if _truthy(_cfg(cfg, "CPU_PARALLEL_STATS_ENABLED", "1"), True) else "0"),
    ("AVABM_FAST_EQUIV_MATH", "1" if _truthy(_cfg(cfg, "CUDA_FAST_EQUIV_MATH", "1"), True) else "0"),
    ("AVABM_USE_ASYNC_MEMSET_CLEAR", "1" if _truthy(_cfg(cfg, "CUDA_USE_ASYNC_MEMSET_CLEAR", "1"), True) else "0"),
    ("AVABM_SPAWN_GRID_INSERT_FASTPATH", "1" if _truthy(_cfg(cfg, "CUDA_SPAWN_GRID_INSERT_FASTPATH", "1"), True) else "0"),
    ("AVABM_SPAWN_ALLOC_SCAN_LIMIT", str(max(32, _int_value(_cfg(cfg, "CUDA_SPAWN_ALLOC_SCAN_LIMIT", "4096"), 4096)))),
    ("AVABM_ECS_CURSOR_ALLOCATOR", "1" if _truthy(_cfg(cfg, "CUDA_ECS_CURSOR_ALLOCATOR", "1"), True) else "0"),
    ("AVABM_INCREMENTAL_ACTIVE_LIST", "1" if _truthy(_cfg(cfg, "CUDA_INCREMENTAL_ACTIVE_LIST", "1"), True) else "0"),
    ("AVABM_ACTIVE_FULL_COMPACT_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_ACTIVE_FULL_COMPACT_INTERVAL", "8"), 8)))),
    ("AVABM_MIN_CRUISE_SPEED_ENABLED", "1" if _truthy(_cfg(cfg, "SPEED_MIN_CRUISE_ENABLED", "1"), True) else "0"),
    ("AVABM_MIN_CRUISE_SPEED_KMH", f"{max(0.0, _float_value(_cfg(cfg, 'SPEED_MIN_CRUISE_KMH', '40.0'), 40.0)):.6f}"),
    ("AVABM_MIN_CRUISE_HARD_FREEFLOW", "1" if _truthy(_cfg(cfg, "SPEED_MIN_CRUISE_HARD_FREEFLOW", "1"), True) else "0"),
    ("AVABM_ACTIVE_LIST_ENABLED", "1" if _truthy(_cfg(cfg, "CUDA_ACTIVE_LIST_ENABLED", "1"), True) else "0"),
    ("AVABM_LAZY_GRID_ENABLED", "1" if _truthy(_cfg(cfg, "CUDA_LAZY_GRID_ENABLED", "1"), True) else "0"),
    ("AVABM_LANE_HASH_GRID_ENABLED", "1" if _truthy(_cfg(cfg, "CUDA_LANE_HASH_GRID_ENABLED", "1" if lane_hash_default else "0"), lane_hash_default) else "0"),
    ("AVABM_FAST_PIPELINE_MODE", "1" if fast_physics_mode else "0"),
    ("AVABM_FAST_PHYSICS_MODE", "1" if fast_physics_mode else "0"),
    ("AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS", str(max(1, _int_value(_cfg(cfg, "CUDA_ACTIVE_KERNEL_BLOCKS", "128"), 128)))),
    ("CONTACT_RESOLVE_PASSES", str(max(1, min(16, _int_value(_cfg(cfg, "CUDA_CONTACT_RESOLVE_PASSES", "9"), 9))))),
    ("AVABM_PERSISTENT_START_GRID", "1" if _truthy(_cfg(cfg, "CUDA_PERSISTENT_START_GRID", "1"), True) else "0"),
    ("AVABM_ECS_PERSISTENT_PERCEPTION_GRID_REUSE", "1" if _truthy(_cfg(cfg, "CUDA_ECS_PERSISTENT_PERCEPTION_GRID_REUSE", "1"), True) else "0"),
    ("AVABM_SKIP_PRESPAWN_ACTIVE_REBUILD", "1" if _truthy(_cfg(cfg, "CUDA_SKIP_PRESPAWN_ACTIVE_REBUILD", "1"), True) else "0"),
    ("AVABM_ROUTE_REPAIR_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_ROUTE_REPAIR_INTERVAL", "1"), 1)))),
    ("AVABM_SPAWN_OVERLAP_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_SPAWN_OVERLAP_INTERVAL", "1"), 1)))),
    ("AVABM_LOCAL_AVOIDANCE_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_LOCAL_AVOIDANCE_INTERVAL", "1"), 1)))),
    ("AVABM_FRONT_CLEAR_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_FRONT_CLEAR_INTERVAL", "1"), 1)))),
    ("AVABM_CONTACT_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_CONTACT_INTERVAL", "1"), 1)))),
    ("AVABM_COLLISION_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_COLLISION_INTERVAL", "1"), 1)))),
    ("AVABM_SAFETY_METRICS_INTERVAL", str(safety_interval)),
    ("AVABM_STATS_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_STATS_INTERVAL", "1"), 1)))),
    ("AVABM_PRIORITY_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_PRIORITY_INTERVAL", "1"), 1)))),
    ("AVABM_LANE_BUCKET_SCAN_LIMIT", str(max(0, _int_value(_cfg(cfg, "CUDA_LANE_BUCKET_SCAN_LIMIT", "0"), 0)))),
    ("AVABM_WORLD_BUCKET_SCAN_LIMIT", str(max(0, _int_value(_cfg(cfg, "CUDA_WORLD_BUCKET_SCAN_LIMIT", "0"), 0)))),
    ("WORLD_MAX_CELL_RADIUS", str(max(1, _int_value(_cfg(cfg, "CUDA_WORLD_MAX_CELL_RADIUS", "5"), 5)))),
    ("CONTACT_CELL_RADIUS", str(max(1, _int_value(_cfg(cfg, "CUDA_CONTACT_CELL_RADIUS", "3"), 3)))),
    ("COLLISION_CELL_RADIUS", str(max(1, _int_value(_cfg(cfg, "CUDA_COLLISION_CELL_RADIUS", "3"), 3)))),
    ("AVABM_TURBO_LOCAL_AVOID_CULL", "1" if _truthy(_cfg(cfg, "CUDA_TURBO_LOCAL_AVOID_CULL", "0"), False) else "0"),
    ("AVABM_TURBO_FAST_NEIGHBOR_SENSORS", "1" if _truthy(_cfg(cfg, "CUDA_TURBO_FAST_NEIGHBOR_SENSORS", "1"), True) else "0"),
    ("AVABM_METRICS_MODE", str(max(0, min(2, _int_value(_cfg(cfg, "CUDA_METRICS_MODE", "1"), 1))))),
    ("AVABM_EXPENSIVE_SAFETY_METRICS_ENABLED", "1" if _truthy(_cfg(cfg, "CUDA_EXPENSIVE_SAFETY_METRICS_ENABLED", "1"), True) else "0"),
    ("AVABM_EXPENSIVE_SAFETY_METRICS_INTERVAL", str(max(1, _int_value(_cfg(cfg, "CUDA_EXPENSIVE_SAFETY_METRICS_INTERVAL", str(safety_interval)), safety_interval)))),
    ("AVABM_DEVICE_FORCEINLINE", "1" if _truthy(_cfg(cfg, "CUDA_DEVICE_FORCEINLINE", "0"), False) else "0"),
    ("AVABM_DEVICE_NOINLINE", "1" if _truthy(_cfg(cfg, "CUDA_DEVICE_NOINLINE", "0"), False) else "0"),
]

if os.name == "nt":
    compile_args = [f"/std:c++{std}", "/EHsc", "/bigobj", "/O2" if opt != "0" else "/Od", "/wd4068", "/wd4505", "/wd4244", "/wd4267", "/wd4018"] + extra
    link_args = []
else:
    opt_flag = f"-O{opt.lower()}" if opt in {"0", "1", "2", "3", "S", "G"} else "-O2"
    compile_args = [f"-std=c++{std}", opt_flag, "-pthread", "-Wno-unused-parameter", "-Wno-unused-function", "-Wno-unknown-pragmas", "-Wno-sign-compare", "-Wno-misleading-indentation"] + extra
    link_args = ["-pthread"]

print(f"[Info] AVABM CUDA-source-parity CPU extension: c++{std}, O{opt}, flags={' '.join(compile_args)}")

_BUILD_RESULT = setup(
    name="avabm",
    version="0.3.0",
    packages=["avabm"],
    package_dir={"avabm": str(PACKAGE_DIR)},
    ext_modules=[
        Extension(
            "avabm.avabm_cpu_ext",
            [str(SRC_DIR / "binding.cpp"), str(SRC_DIR / "cpu_kernels.cpp")],
            include_dirs=[str(SRC_DIR), str(SRC_DIR.parent / "cuda")],
            language="c++",
            extra_compile_args=compile_args,
            extra_link_args=link_args,
            define_macros=defines,
        )
    ],
)

def _source_fingerprint():
    h = hashlib.sha256()
    for p in sorted([SRC_DIR / "binding.cpp", SRC_DIR / "cpu_kernels.cpp", SRC_DIR / "cpu_port_api.hpp", SRC_DIR / "main_common_cpu.hpp", SRC_DIR / "setup.py"], key=lambda x: str(x)):
        if not p.exists():
            continue
        try:
            rel = str(p.relative_to(ROOT)).replace(os.sep, "/")
        except Exception:
            rel = str(p)
        h.update(rel.encode("utf-8", "ignore"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()

try:
    (SRC_DIR / ".avabm_cpu_ext_fingerprint").write_text(_source_fingerprint() + "\n", encoding="utf-8")
except Exception as e:
    print(f"[Warning] Failed to write CPU extension fingerprint: {e}")
