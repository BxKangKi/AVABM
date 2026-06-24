from pathlib import Path
import os
import re
import shlex
import subprocess
from setuptools import setup
def _read_config(path: Path):
    cfg = {}
    if not path.exists():
        return cfg
    for raw in path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.split("#", 1)[0].split(";", 1)[0].strip().strip('"').strip("'")
        if key:
            cfg[key] = None if value.lower() in {"", "none", "null"} else os.path.expandvars(value)
    return cfg
def _truthy(value, default=False):
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "none", ""}
def _int_value(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)
def _float_value(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
def _split_flags(value):
    if not value:
        return []
    return shlex.split(str(value), posix=(os.name != "nt"))
ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG = _read_config(ROOT_DIR / "config.txt")
def _cfg(key, default=None):
    value = os.environ.get(key)
    if value is not None and str(value).strip() != "":
        return os.path.expandvars(str(value).strip().strip('"').strip("\'"))
    return CONFIG.get(key, default)
def _nvcc_exe_name() -> str:
    return "nvcc.exe" if os.name == "nt" else "nvcc"
def _path_has_nvcc(path_value: str | None) -> bool:
    if not path_value:
        return False
    try:
        return (Path(path_value) / "bin" / _nvcc_exe_name()).exists()
    except OSError:
        return False
def _candidate_cuda_homes():
    seen = set()
    def add(value):
        if not value:
            return
        text = os.path.expandvars(str(value).strip().strip('"'))
        if not text or text.lower() in {"auto", "latest", "detect"}:
            return
        key = os.path.normcase(os.path.abspath(text))
        if key in seen:
            return
        seen.add(key)
        yield text
    for name in ("CUDA_HOME", "CUDA_PATH"):
        yield from add(os.environ.get(name))
    for name in sorted(os.environ):
        if name.startswith("CUDA_PATH_V"):
            yield from add(os.environ.get(name))
    if os.name == "nt":
        bases = [
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            os.environ.get("ProgramW6432", r"C:\Program Files"),
        ]
        versions = [
            "v13.3", "v13.2", "v13.1", "v13.0",
            "v12.9", "v12.8", "v12.7", "v12.6", "v12.5", "v12.4",
        ]
        for base in bases:
            cuda_root = Path(base) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            for version in versions:
                yield from add(str(cuda_root / version))
            try:
                dirs = sorted(cuda_root.glob("v*"), reverse=True)
            except OSError:
                dirs = []
            for path in dirs:
                yield from add(str(path))
    else:
        for value in (
            "/usr/local/cuda",
            "/usr/local/cuda-13.3",
            "/usr/local/cuda-13.2",
            "/usr/local/cuda-13.1",
            "/usr/local/cuda-13.0",
            "/usr/local/cuda-12.9",
            "/usr/local/cuda-12.8",
            "/usr/local/cuda-12.7",
            "/usr/local/cuda-12.6",
        ):
            yield from add(value)
def _resolve_cuda_home() -> str | None:
    requested = os.environ.get("CUDA_HOME") or _cfg("CUDA_HOME") or "auto"
    requested_text = str(requested).strip() if requested is not None else "auto"
    if requested_text and requested_text.lower() not in {"auto", "latest", "detect"}:
        explicit = os.path.expandvars(requested_text)
        if _path_has_nvcc(explicit):
            return explicit
        print(f"[Warning] CUDA_HOME does not contain nvcc and will be auto-detected instead: {explicit}")
    for candidate in _candidate_cuda_homes():
        if _path_has_nvcc(candidate):
            return candidate
    return None
if _truthy(_cfg("CUDA_CLEAR_NVCC_GLOBAL_FLAGS"), default=True):
    os.environ.pop("NVCC_PREPEND_FLAGS", None)
    os.environ.pop("NVCC_APPEND_FLAGS", None)
_resolved_cuda_home = _resolve_cuda_home()
if _resolved_cuda_home:
    os.environ["CUDA_HOME"] = _resolved_cuda_home
    print(f"[Info] CUDA_HOME resolved to {_resolved_cuda_home}")
elif _cfg("CUDA_HOME") and str(_cfg("CUDA_HOME")).strip().lower() not in {"auto", "latest", "detect"}:
    os.environ["CUDA_HOME"] = str(CONFIG["CUDA_HOME"])
for key in (
    "MAX_JOBS",
    "TORCH_NVCC_FLAGS",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_NATIVE_GPU_ONLY",
    "CUDA_ARCH_FALLBACK",
    "CUDA_BUILD_MODE",
    "CUDA_NVCC_THREADS",
    "CUDA_SPLIT_COMPILE",
    "CUDA_ARCH_KEEP_PTX",
    "CUDA_PTXAS_SAFE_MODE",
    "CUDA_PTXAS_OPT_LEVEL",
    "CUDA_PTXAS_ALLOW_EXPENSIVE_OPT",
    "CUDA_NVCC_TIME_LOG",
    "TORCH_DONT_CHECK_COMPILER_ABI",
):
    if _cfg(key) and not os.environ.get(key):
        os.environ[key] = CONFIG[key]
if not os.environ.get("TORCH_DONT_CHECK_COMPILER_ABI"):
    os.environ["TORCH_DONT_CHECK_COMPILER_ABI"] = "1"
def _detect_arch_with_nvidia_smi() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    for line in out.splitlines():
        m = re.search(r"(\d+)\s*\.\s*(\d+)", line)
        if m:
            return f"{int(m.group(1))}.{int(m.group(2))}"
    return None
def _detect_arch_with_torch() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{int(major)}.{int(minor)}"
    except Exception:
        return None
    return None
def _normalize_arch_part(part: str) -> str:
    item = str(part).strip().replace("+PTX", "").replace("+ptx", "")
    m = re.search(r"(\d+)\s*\.\s*(\d+)", item)
    if m:
        return f"{int(m.group(1))}.{int(m.group(2))}"
    m = re.search(r"(?:sm_|compute_)(\d+)", item, re.IGNORECASE)
    if m:
        digits = m.group(1)
        if len(digits) >= 3:
            return f"{int(digits[:-1])}.{int(digits[-1])}"
        if len(digits) == 2:
            return f"{int(digits[0])}.{int(digits[1])}"
    return item
def _arch_to_sm_code(arch: str) -> str | None:
    m = re.search(r"(\d+)\s*\.\s*(\d+)", str(arch))
    if not m:
        return None
    return f"sm_{int(m.group(1))}{int(m.group(2))}"
def _resolve_native_cuda_arch() -> str:
    native_only = _truthy(_cfg("CUDA_NATIVE_GPU_ONLY"), default=True)
    keep_ptx = _truthy(_cfg("CUDA_ARCH_KEEP_PTX"), default=False)
    arch_value = str(os.environ.get("TORCH_CUDA_ARCH_LIST") or _cfg("TORCH_CUDA_ARCH_LIST") or "auto").strip()
    fallback = str(_cfg("CUDA_ARCH_FALLBACK") or "8.6").strip()
    needs_auto = (not arch_value) or arch_value.lower() in {"auto", "native", "gpu", "local"}
    if not native_only:
        resolved = (_detect_arch_with_nvidia_smi() or _detect_arch_with_torch() or fallback) if needs_auto else arch_value
        os.environ["TORCH_CUDA_ARCH_LIST"] = resolved
        print(f"[Info] TORCH_CUDA_ARCH_LIST resolved to {resolved} (native GPU only=0)")
        return resolved
    if needs_auto:
        resolved = _detect_arch_with_nvidia_smi() or _detect_arch_with_torch() or fallback
    else:
        resolved = arch_value
    parts = []
    for part in str(resolved).replace(",", ";").split(";"):
        item = _normalize_arch_part(part)
        if item and item not in parts:
            parts.append(item)
    if not parts:
        parts = [_normalize_arch_part(fallback)]
    resolved = parts[0]
    if keep_ptx:
        resolved = f"{resolved}+PTX"
    os.environ["TORCH_CUDA_ARCH_LIST"] = resolved
    print(f"[Info] TORCH_CUDA_ARCH_LIST resolved to {resolved} (native GPU only=1, keep PTX={int(keep_ptx)})")
    return resolved
def _nvcc_supported_sm_codes() -> set[str]:
    cuda_home = os.environ.get("CUDA_HOME") or _resolved_cuda_home
    nvcc = Path(cuda_home) / "bin" / _nvcc_exe_name() if cuda_home else Path(_nvcc_exe_name())
    try:
        out = subprocess.check_output([str(nvcc), "--list-gpu-code"], stderr=subprocess.STDOUT, text=True, timeout=12)
    except Exception:
        return set()
    return set(re.findall(r"sm_\d+", out))
def _validate_resolved_arch_with_nvcc(arch_list: str) -> None:
    if not _truthy(_cfg("CUDA_VALIDATE_NVCC_ARCH"), default=True):
        return
    supported = _nvcc_supported_sm_codes()
    if not supported:
        return
    missing = []
    for part in str(arch_list).replace(",", ";").split(";"):
        clean = _normalize_arch_part(part)
        sm = _arch_to_sm_code(clean)
        if sm and sm not in supported:
            missing.append(sm)
    if missing:
        raise RuntimeError(
            "Resolved TORCH_CUDA_ARCH_LIST targets "
            + ", ".join(missing)
            + ", but the selected nvcc does not list that GPU code. "
            + "Set CUDA_HOME=auto or install/select a CUDA Toolkit new enough for this GPU. "
            + f"Current CUDA_HOME={os.environ.get('CUDA_HOME', '<not set>')}"
        )
_RESOLVED_CUDA_ARCH_LIST = _resolve_native_cuda_arch()
_validate_resolved_arch_with_nvcc(_RESOLVED_CUDA_ARCH_LIST)
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
build_mode = str(_cfg("CUDA_BUILD_MODE", "release")).strip().lower()
if build_mode not in {"fastdev", "release", "benchmark", "debug"}:
    build_mode = "fastdev"
default_opt_level = {"fastdev": "1", "release": "2", "benchmark": "3", "debug": "0"}.get(build_mode, "1")
opt_level = str(_cfg("CUDA_OPT_LEVEL", default_opt_level)).strip().upper().lstrip("O")
if opt_level not in {"0", "1", "2", "3"}:
    opt_level = default_opt_level
cxx_standard = str(_cfg("CUDA_CXX_STANDARD", "17")).strip().lower().replace("c++", "")
if cxx_standard not in {"17", "20"}:
    cxx_standard = "17"
show_warnings = _truthy(_cfg("CUDA_SHOW_WARNINGS"), default=False)
suppress_header_warnings = _truthy(_cfg("CUDA_SUPPRESS_HEADER_WARNINGS"), default=True)
fast_math = _truthy(_cfg("CUDA_FAST_MATH"), default=False)
nvcc_threads = max(0, _int_value(_cfg("CUDA_NVCC_THREADS"), default=0))
split_compile = max(0, _int_value(_cfg("CUDA_SPLIT_COMPILE"), default=0))
ptxas_safe_mode = _truthy(_cfg("CUDA_PTXAS_SAFE_MODE"), default=False)
disable_msvc_gl = _truthy(_cfg("CUDA_DISABLE_MSVC_GL"), default=True)
use_full_torch_header = _truthy(_cfg("CUDA_USE_FULL_TORCH_EXTENSION_HEADER"), default=False)
fast_equiv_math = _truthy(_cfg("CUDA_FAST_EQUIV_MATH"), default=True)
use_async_memset_clear = _truthy(_cfg("CUDA_USE_ASYNC_MEMSET_CLEAR"), default=True)
spawn_grid_insert_fastpath = _truthy(_cfg("CUDA_SPAWN_GRID_INSERT_FASTPATH"), default=True)
spawn_alloc_scan_limit = max(32, _int_value(_cfg("CUDA_SPAWN_ALLOC_SCAN_LIMIT"), default=4096))
ecs_cursor_allocator = _truthy(_cfg("CUDA_ECS_CURSOR_ALLOCATOR"), default=True)
incremental_active_list = _truthy(_cfg("CUDA_INCREMENTAL_ACTIVE_LIST"), default=True)
active_full_compact_interval = max(1, _int_value(_cfg("CUDA_ACTIVE_FULL_COMPACT_INTERVAL"), default=8))
min_cruise_enabled = _truthy(_cfg("SPEED_MIN_CRUISE_ENABLED"), default=True)
min_cruise_kmh = max(0.0, _float_value(_cfg("SPEED_MIN_CRUISE_KMH"), default=40.0))
min_cruise_hard_freeflow = _truthy(_cfg("SPEED_MIN_CRUISE_HARD_FREEFLOW"), default=True)
active_list_enabled = _truthy(_cfg("CUDA_ACTIVE_LIST_ENABLED"), default=True)
lazy_grid_enabled = _truthy(_cfg("CUDA_LAZY_GRID_ENABLED"), default=True)
_lane_hash_default = _truthy(_cfg("CUDA_FAST_PHYSICS_MODE"), default=False)
lane_hash_grid_enabled = _truthy(_cfg("CUDA_LANE_HASH_GRID_ENABLED"), default=_lane_hash_default)
active_kernel_blocks = max(1, _int_value(_cfg("CUDA_ACTIVE_KERNEL_BLOCKS"), default=128))
contact_resolve_passes = max(1, min(16, _int_value(_cfg("CUDA_CONTACT_RESOLVE_PASSES"), default=9)))
persistent_start_grid = _truthy(_cfg("CUDA_PERSISTENT_START_GRID"), default=True)
persistent_perception_grid_reuse = _truthy(_cfg("CUDA_ECS_PERSISTENT_PERCEPTION_GRID_REUSE"), default=True)
fast_physics_mode = _truthy(_cfg("CUDA_FAST_PHYSICS_MODE"), default=False)
skip_prespawn_active_rebuild = _truthy(_cfg("CUDA_SKIP_PRESPAWN_ACTIVE_REBUILD"), default=True)
route_repair_interval = max(1, _int_value(_cfg("CUDA_ROUTE_REPAIR_INTERVAL"), default=1))
spawn_overlap_interval = max(1, _int_value(_cfg("CUDA_SPAWN_OVERLAP_INTERVAL"), default=1))
local_avoidance_interval = max(1, _int_value(_cfg("CUDA_LOCAL_AVOIDANCE_INTERVAL"), default=1))
front_clear_interval = max(1, _int_value(_cfg("CUDA_FRONT_CLEAR_INTERVAL"), default=1))
contact_interval = max(1, _int_value(_cfg("CUDA_CONTACT_INTERVAL"), default=1))
collision_interval = max(1, _int_value(_cfg("CUDA_COLLISION_INTERVAL"), default=1))
safety_metrics_interval = max(1, _int_value(_cfg("CUDA_SAFETY_METRICS_INTERVAL"), default=1))
stats_interval = max(1, _int_value(_cfg("CUDA_STATS_INTERVAL"), default=1))
priority_interval = max(1, _int_value(_cfg("CUDA_PRIORITY_INTERVAL"), default=1))
lane_bucket_scan_limit = max(0, _int_value(_cfg("CUDA_LANE_BUCKET_SCAN_LIMIT"), default=0))
world_bucket_scan_limit = max(0, _int_value(_cfg("CUDA_WORLD_BUCKET_SCAN_LIMIT"), default=0))
world_max_cell_radius = max(1, _int_value(_cfg("CUDA_WORLD_MAX_CELL_RADIUS"), default=5))
contact_cell_radius = max(1, _int_value(_cfg("CUDA_CONTACT_CELL_RADIUS"), default=3))
collision_cell_radius = max(1, _int_value(_cfg("CUDA_COLLISION_CELL_RADIUS"), default=3))
turbo_local_avoid_cull = _truthy(_cfg("CUDA_TURBO_LOCAL_AVOID_CULL"), default=False)
turbo_fast_neighbor_sensors = _truthy(_cfg("CUDA_TURBO_FAST_NEIGHBOR_SENSORS"), default=True)
metrics_mode = max(0, min(2, _int_value(_cfg("CUDA_METRICS_MODE"), default=1)))
expensive_safety_metrics_enabled = _truthy(_cfg("CUDA_EXPENSIVE_SAFETY_METRICS_ENABLED"), default=True)
expensive_safety_metrics_interval = max(1, _int_value(_cfg("CUDA_EXPENSIVE_SAFETY_METRICS_INTERVAL"), default=safety_metrics_interval))
device_forceinline = _truthy(_cfg("CUDA_DEVICE_FORCEINLINE"), default=False)
device_noinline = _truthy(_cfg("CUDA_DEVICE_NOINLINE"), default=False)
binding_opt_level = str(_cfg("CUDA_BINDING_OPT_LEVEL", "1")).strip().upper().lstrip("O")
if binding_opt_level not in {"0", "1", "2"}:
    binding_opt_level = "0"
ptxas_default_opt = "0" if ptxas_safe_mode else ("1" if build_mode == "fastdev" else ("3" if build_mode == "benchmark" else "2"))
ptxas_opt_level = str(_cfg("CUDA_PTXAS_OPT_LEVEL", ptxas_default_opt)).strip().upper().lstrip("O")
if ptxas_opt_level not in {"0", "1", "2", "3"}:
    ptxas_opt_level = ptxas_default_opt
ptxas_allow_expensive = _truthy(
    _cfg("CUDA_PTXAS_ALLOW_EXPENSIVE_OPT"),
    default=(not ptxas_safe_mode and build_mode in {"benchmark"} and ptxas_opt_level in {"2", "3"}),
)
if ptxas_safe_mode:
    if nvcc_threads > 1:
        print(f"[Info] CUDA_PTXAS_SAFE_MODE=1: ignoring CUDA_NVCC_THREADS={nvcc_threads}; serial nvcc is safer for large sm_86 builds.")
        nvcc_threads = 0
    if split_compile > 1:
        print(f"[Info] CUDA_PTXAS_SAFE_MODE=1: ignoring CUDA_SPLIT_COMPILE={split_compile}; split-compile can destabilize ptxas on some systems.")
        split_compile = 0
    if int(ptxas_opt_level) > 0:
        print(f"[Info] CUDA_PTXAS_SAFE_MODE=1: lowering ptxas optimization O{ptxas_opt_level} -> O0.")
        ptxas_opt_level = "0"
    ptxas_allow_expensive = False
nvcc_time_log = _truthy(_cfg("CUDA_NVCC_TIME_LOG"), default=True)
add_nvcc_std_flag = _truthy(_cfg("CUDA_NVCC_ADD_STD_FLAG"), default=False)
nvcc_flags = [f"-O{opt_level}", "--expt-relaxed-constexpr"]
if add_nvcc_std_flag:
    nvcc_flags.append(f"-std=c++{cxx_standard}")
nvcc_flags.extend(["-Xptxas", f"-O{ptxas_opt_level}"])
nvcc_flags.extend(["-Xptxas", f"-allow-expensive-optimizations={'true' if ptxas_allow_expensive else 'false'}"])
if nvcc_time_log:
    nvcc_time_path = ROOT_DIR / "avabm_cuda" / "build_logs" / "nvcc_time.csv"
    nvcc_time_path.parent.mkdir(exist_ok=True)
    nvcc_flags.extend(["--time", str(nvcc_time_path)])
if fast_math:
    nvcc_flags.append("--use_fast_math")
if nvcc_threads > 1:
    nvcc_flags.extend(["--threads", str(nvcc_threads)])
if split_compile > 1:
    nvcc_flags.extend(["--split-compile", str(split_compile)])
if not show_warnings:
    nvcc_flags.extend(["-w"])
for flag in _split_flags(os.environ.get("TORCH_NVCC_FLAGS") or _cfg("TORCH_NVCC_FLAGS")):
    if flag and flag not in nvcc_flags:
        nvcc_flags.append(flag)
for flag in _split_flags(_cfg("CUDA_EXTRA_NVCC_FLAGS")):
    nvcc_flags.append(flag)
if os.name == "nt":
    if binding_opt_level in {"1", "2"}:
        cxx_flags = [f"/O{binding_opt_level}"]
    else:
        cxx_flags = ["/Od", "/Ob0"]
    if disable_msvc_gl:
        cxx_flags.append("/GL-")
    cxx_flags.extend([f"/std:c++{cxx_standard}", "/EHsc", "/bigobj"])
    if suppress_header_warnings:
        warning_disables = [
            "/wd4996", "/wd4819", "/wd4251", "/wd4275", "/wd4244", "/wd4267",
            "/wd4018", "/wd4190", "/wd4624", "/wd4067", "/wd4068", "/wd4273",
        ]
        cxx_flags.extend(warning_disables)
        for wd in warning_disables:
            nvcc_flags.extend(["-Xcompiler", wd])
else:
    cxx_flags = [f"-O{binding_opt_level}" if binding_opt_level in {"1", "2"} else "-O0", f"-std=c++{cxx_standard}"]
    if suppress_header_warnings:
        cxx_flags.extend(["-Wno-deprecated-declarations", "-Wno-unused-parameter"])
        nvcc_flags.append("-Xcompiler=-Wno-deprecated-declarations")
for flag in _split_flags(_cfg("CUDA_EXTRA_CXX_FLAGS")):
    cxx_flags.append(flag)
print(
    "[Info] AVABM CUDA compile profile: "
    f"mode={build_mode}, nvcc_O{opt_level}, ptxas_O{ptxas_opt_level}, "
    f"ptxas_safe={int(ptxas_safe_mode)}, split_compile={split_compile}, nvcc_threads={nvcc_threads}, "
    f"fast_math={int(fast_math)}, contact_passes={contact_resolve_passes}, "
    f"ecs_cursor_alloc={int(ecs_cursor_allocator)}, incremental_active={int(incremental_active_list)}/{active_full_compact_interval}, "
    f"persistent_grid={int(persistent_start_grid)}, perception_grid_reuse={int(persistent_perception_grid_reuse)}, fast_physics={int(fast_physics_mode)}, lane_hash={int(lane_hash_grid_enabled)}, "
    f"skip_prespawn_active={int(skip_prespawn_active_rebuild)}, route_repair_interval={route_repair_interval}, "
    f"priority_interval={priority_interval}, "
    f"contact_interval={contact_interval}, collision_interval={collision_interval}, stats_interval={stats_interval}, "
    f"world_r={world_max_cell_radius}, contact_r={contact_cell_radius}, collision_r={collision_cell_radius}, "
    f"lane_bucket_limit={lane_bucket_scan_limit}, world_bucket_limit={world_bucket_scan_limit}, "
    f"metrics_mode={metrics_mode}, fast_sensors={int(turbo_fast_neighbor_sensors)}, local_cull={int(turbo_local_avoid_cull)}, "
    f"expensive_safety_metrics={int(expensive_safety_metrics_enabled)}/{expensive_safety_metrics_interval}, "
    f"device_noinline={int(device_noinline)}"
)
common_defines = [
    ("_CRT_SECURE_NO_WARNINGS", None),
    ("_SCL_SECURE_NO_WARNINGS", None),
    ("_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS", None),
    ("AVABM_CUDA_BUILD_V41", "1"),
]
if use_full_torch_header:
    common_defines.append(("AVABM_USE_FULL_TORCH_EXTENSION_HEADER", "1"))
common_defines.extend([
    ("AVABM_FAST_EQUIV_MATH", "1" if fast_equiv_math else "0"),
    ("AVABM_USE_ASYNC_MEMSET_CLEAR", "1" if use_async_memset_clear else "0"),
    ("AVABM_SPAWN_GRID_INSERT_FASTPATH", "1" if spawn_grid_insert_fastpath else "0"),
    ("AVABM_SPAWN_ALLOC_SCAN_LIMIT", str(spawn_alloc_scan_limit)),
    ("AVABM_ECS_CURSOR_ALLOCATOR", "1" if ecs_cursor_allocator else "0"),
    ("AVABM_INCREMENTAL_ACTIVE_LIST", "1" if incremental_active_list else "0"),
    ("AVABM_ACTIVE_FULL_COMPACT_INTERVAL", str(active_full_compact_interval)),
    ("AVABM_MIN_CRUISE_SPEED_ENABLED", "1" if min_cruise_enabled else "0"),
    ("AVABM_MIN_CRUISE_SPEED_KMH", f"{min_cruise_kmh:.6f}"),
    ("AVABM_MIN_CRUISE_HARD_FREEFLOW", "1" if min_cruise_hard_freeflow else "0"),
    ("AVABM_ACTIVE_LIST_ENABLED", "1" if active_list_enabled else "0"),
    ("AVABM_LAZY_GRID_ENABLED", "1" if lazy_grid_enabled else "0"),
    ("AVABM_LANE_HASH_GRID_ENABLED", "1" if lane_hash_grid_enabled else "0"),
    ("AVABM_FAST_PIPELINE_MODE", "1" if fast_physics_mode else "0"),
    ("AVABM_ACTIVE_ENTITY_KERNEL_BLOCKS", str(active_kernel_blocks)),
    ("CONTACT_RESOLVE_PASSES", str(contact_resolve_passes)),
    ("AVABM_PERSISTENT_START_GRID", "1" if persistent_start_grid else "0"),
    ("AVABM_ECS_PERSISTENT_PERCEPTION_GRID_REUSE", "1" if persistent_perception_grid_reuse else "0"),
    ("AVABM_FAST_PHYSICS_MODE", "1" if fast_physics_mode else "0"),
    ("AVABM_SKIP_PRESPAWN_ACTIVE_REBUILD", "1" if skip_prespawn_active_rebuild else "0"),
    ("AVABM_ROUTE_REPAIR_INTERVAL", str(route_repair_interval)),
    ("AVABM_SPAWN_OVERLAP_INTERVAL", str(spawn_overlap_interval)),
    ("AVABM_LOCAL_AVOIDANCE_INTERVAL", str(local_avoidance_interval)),
    ("AVABM_FRONT_CLEAR_INTERVAL", str(front_clear_interval)),
    ("AVABM_CONTACT_INTERVAL", str(contact_interval)),
    ("AVABM_COLLISION_INTERVAL", str(collision_interval)),
    ("AVABM_SAFETY_METRICS_INTERVAL", str(safety_metrics_interval)),
    ("AVABM_STATS_INTERVAL", str(stats_interval)),
    ("AVABM_PRIORITY_INTERVAL", str(priority_interval)),
    ("AVABM_LANE_BUCKET_SCAN_LIMIT", str(lane_bucket_scan_limit)),
    ("AVABM_WORLD_BUCKET_SCAN_LIMIT", str(world_bucket_scan_limit)),
    ("WORLD_MAX_CELL_RADIUS", str(world_max_cell_radius)),
    ("CONTACT_CELL_RADIUS", str(contact_cell_radius)),
    ("COLLISION_CELL_RADIUS", str(collision_cell_radius)),
    ("AVABM_TURBO_LOCAL_AVOID_CULL", "1" if turbo_local_avoid_cull else "0"),
    ("AVABM_TURBO_FAST_NEIGHBOR_SENSORS", "1" if turbo_fast_neighbor_sensors else "0"),
    ("AVABM_METRICS_MODE", str(metrics_mode)),
    ("AVABM_EXPENSIVE_SAFETY_METRICS_ENABLED", "1" if expensive_safety_metrics_enabled else "0"),
    ("AVABM_EXPENSIVE_SAFETY_METRICS_INTERVAL", str(expensive_safety_metrics_interval)),
    ("AVABM_DEVICE_FORCEINLINE", "1" if device_forceinline else "0"),
    ("AVABM_DEVICE_NOINLINE", "1" if device_noinline else "0"),
])
_BaseBuildExtension = BuildExtension.with_options(use_ninja=True)
class AVABMBuildExtension(_BaseBuildExtension):
    """BuildExtension with Windows defaults adjusted for reproducible local builds."""
    def _strip_msvc_slow_defaults(self):
        if os.name != "nt":
            return
        remove_compile = {"/gl"}
        if binding_opt_level != "2":
            remove_compile.add("/o2")
        remove_link = {"/ltcg"}
        for attr in ("compile_options", "compile_options_debug"):
            opts = getattr(self.compiler, attr, None)
            if isinstance(opts, list):
                setattr(self.compiler, attr, [opt for opt in opts if str(opt).strip().lower() not in remove_compile])
        for attr in ("ldflags_shared", "ldflags_shared_debug"):
            opts = getattr(self.compiler, attr, None)
            if isinstance(opts, list):
                setattr(self.compiler, attr, [opt for opt in opts if str(opt).strip().lower() not in remove_link])
    def build_extensions(self):
        if os.name == "nt" and disable_msvc_gl and getattr(self, "compiler", None) is not None:
            original_initialize = getattr(self.compiler, "initialize", None)
            if callable(original_initialize) and not getattr(self.compiler, "_avabm_init_wrapped", False):
                def initialize_and_strip(*args, **kwargs):
                    result = original_initialize(*args, **kwargs)
                    self._strip_msvc_slow_defaults()
                    return result
                self.compiler.initialize = initialize_and_strip
                self.compiler._avabm_init_wrapped = True
            if not getattr(self.compiler, "initialized", False) and callable(getattr(self.compiler, "initialize", None)):
                self.compiler.initialize()
            self._strip_msvc_slow_defaults()
        super().build_extensions()
setup(
    name="avabm_cuda",
    ext_modules=[
        CUDAExtension(
            name="avabm_cuda",
            sources=["binding.cpp", "main.cu"],
            define_macros=common_defines,
            extra_compile_args={
                "cxx": cxx_flags,
                "nvcc": nvcc_flags,
            },
        )
    ],
    cmdclass={"build_ext": AVABMBuildExtension},
)
