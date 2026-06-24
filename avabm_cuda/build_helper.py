"""Build helper for the AVABM CUDA extension.
The helper keeps normal runs free of runtime JIT builds: run.bat only verifies
that a matching prebuilt .pyd exists.  Compiling remains an explicit build.bat
step, with a safe retry path for ptxas crashes on large Ampere builds.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
import tokenize
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
CUDA_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUDA_DIR.parent
STAMP_PATH = CUDA_DIR / ".avabm_cuda_build_fingerprint.json"
LAST_BUILD_ENV_PATH = CUDA_DIR / ".avabm_cuda_last_build_env.json"
SOURCE_FILES = [CUDA_DIR / "binding.cpp", CUDA_DIR / "setup.py", CUDA_DIR / "pyproject.toml", CUDA_DIR / "main.cu"]
BUILD_CONFIG_KEYS = [
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "CUDA_NATIVE_GPU_ONLY",
    "CUDA_ARCH_FALLBACK",
    "CUDA_ARCH_KEEP_PTX",
    "CUDA_VALIDATE_NVCC_ARCH",
    "TORCH_NVCC_FLAGS",
    "TORCH_DONT_CHECK_COMPILER_ABI",
    "CUDA_EXTRA_NVCC_FLAGS",
    "CUDA_EXTRA_CXX_FLAGS",
    "CUDA_SUPPRESS_HEADER_WARNINGS",
    "CUDA_SHOW_WARNINGS",
    "CUDA_BUILD_MODE",
    "CUDA_CXX_STANDARD",
    "CUDA_NVCC_ADD_STD_FLAG",
    "CUDA_DISABLE_MSVC_GL",
    "CUDA_BINDING_OPT_LEVEL",
    "CUDA_USE_FULL_TORCH_EXTENSION_HEADER",
    "CUDA_OPT_LEVEL",
    "CUDA_NVCC_TIME_LOG",
    "CUDA_PTXAS_SAFE_MODE",
    "CUDA_PTXAS_ALLOW_EXPENSIVE_OPT",
    "CUDA_PTXAS_OPT_LEVEL",
    "CUDA_FAST_MATH",
    "CUDA_FAST_EQUIV_MATH",
    "CUDA_USE_ASYNC_MEMSET_CLEAR",
    "CUDA_SPAWN_GRID_INSERT_FASTPATH",
    "CUDA_SPAWN_ALLOC_SCAN_LIMIT",
    "CUDA_ECS_CURSOR_ALLOCATOR",
    "CUDA_INCREMENTAL_ACTIVE_LIST",
    "CUDA_ACTIVE_FULL_COMPACT_INTERVAL",
    "CUDA_ACTIVE_LIST_ENABLED",
    "CUDA_ACTIVE_KERNEL_BLOCKS",
    "CUDA_LAZY_GRID_ENABLED",
    "CUDA_LANE_HASH_GRID_ENABLED",
    "CUDA_PERSISTENT_START_GRID",
    "CUDA_ECS_PERSISTENT_PERCEPTION_GRID_REUSE",
    "CUDA_CONTACT_RESOLVE_PASSES",
    "CUDA_ROUTE_REPAIR_INTERVAL",
    "CUDA_EXPENSIVE_SAFETY_METRICS_ENABLED",
    "CUDA_EXPENSIVE_SAFETY_METRICS_INTERVAL",
    "CUDA_FAST_PHYSICS_MODE",
    "CUDA_SKIP_PRESPAWN_ACTIVE_REBUILD",
    "CUDA_SPAWN_OVERLAP_INTERVAL",
    "CUDA_LOCAL_AVOIDANCE_INTERVAL",
    "CUDA_FRONT_CLEAR_INTERVAL",
    "CUDA_CONTACT_INTERVAL",
    "CUDA_COLLISION_INTERVAL",
    "CUDA_SAFETY_METRICS_INTERVAL",
    "CUDA_STATS_INTERVAL",
    "CUDA_PRIORITY_INTERVAL",
    "CUDA_LANE_BUCKET_SCAN_LIMIT",
    "CUDA_WORLD_BUCKET_SCAN_LIMIT",
    "CUDA_WORLD_MAX_CELL_RADIUS",
    "CUDA_CONTACT_CELL_RADIUS",
    "CUDA_COLLISION_CELL_RADIUS",
    "CUDA_TURBO_LOCAL_AVOID_CULL",
    "CUDA_TURBO_FAST_NEIGHBOR_SENSORS",
    "CUDA_METRICS_MODE",
    "CUDA_DEVICE_FORCEINLINE",
    "CUDA_DEVICE_NOINLINE",
    "SPEED_MIN_CRUISE_ENABLED",
    "SPEED_MIN_CRUISE_KMH",
    "SPEED_MIN_CRUISE_HARD_FREEFLOW",
    "CUDA_NVCC_THREADS",
    "CUDA_SPLIT_COMPILE",
    "CUDA_AUTO_RETRY_PTXAS_CRASH",
    "CUDA_SAFE_RETRY_CLEAN",
    "CUDA_CLEAR_NVCC_GLOBAL_FLAGS",
    "MSVC_TOOLSET",
    "MSVC_TOOLSET_VERSION",
    "MSVC_STRICT_TOOLSET",
    "MSVC_VCVARS_VER",
    "MSVC_VCVARS_ARGS",
    "MSVC_VCVARS64_BAT",
]
ACTUAL_ENV_OVERRIDE_KEYS = {
    "CUDA_BUILD_MODE",
    "CUDA_OPT_LEVEL",
    "CUDA_PTXAS_SAFE_MODE",
    "CUDA_PTXAS_OPT_LEVEL",
    "CUDA_PTXAS_ALLOW_EXPENSIVE_OPT",
    "CUDA_FAST_MATH",
    "CUDA_DEVICE_FORCEINLINE",
    "CUDA_DEVICE_NOINLINE",
    "CUDA_NVCC_THREADS",
    "CUDA_SPLIT_COMPILE",
    "CUDA_NVCC_ADD_STD_FLAG",
    "MAX_JOBS",
    "CUDA_BUILD_MAX_JOBS",
}
def _norm_text(value: str | None) -> str:
    return "" if value is None else str(value).strip()
def _truthy(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() not in {"0", "false", "no", "off", "none", ""}
def read_config() -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    path = ROOT_DIR / "config.txt"
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
            cfg[key] = os.path.expandvars(value)
    return cfg
def _nvcc_exe_name() -> str:
    return "nvcc.exe" if os.name == "nt" else "nvcc"
def _path_has_nvcc(path_value: str | None) -> bool:
    if not path_value:
        return False
    try:
        return (Path(path_value) / "bin" / _nvcc_exe_name()).exists()
    except OSError:
        return False
def _candidate_cuda_homes(cfg: Dict[str, str]):
    seen: set[str] = set()
    def add(value: str | None):
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
    yield from add(os.environ.get("CUDA_HOME"))
    yield from add(cfg.get("CUDA_HOME"))
    yield from add(os.environ.get("CUDA_PATH"))
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
def resolve_cuda_home(cfg: Dict[str, str] | None = None) -> str:
    cfg = cfg or read_config()
    requested = os.environ.get("CUDA_HOME") or cfg.get("CUDA_HOME") or "auto"
    requested_text = str(requested).strip()
    if requested_text and requested_text.lower() not in {"auto", "latest", "detect"}:
        explicit = os.path.expandvars(requested_text)
        if _path_has_nvcc(explicit):
            return explicit
        print(f"[Warning] CUDA_HOME does not contain nvcc and will be auto-detected instead: {explicit}")
    for candidate in _candidate_cuda_homes(cfg):
        if _path_has_nvcc(candidate):
            return candidate
    return ""
def nvcc_path_for(cfg: Dict[str, str] | None = None) -> Path:
    cuda_home = resolve_cuda_home(cfg)
    if cuda_home:
        return Path(cuda_home) / "bin" / _nvcc_exe_name()
    return Path(_nvcc_exe_name())
def nvcc_version_text(cfg: Dict[str, str] | None = None) -> str:
    try:
        out = subprocess.check_output([str(nvcc_path_for(cfg)), "-V"], stderr=subprocess.STDOUT, text=True, timeout=8)
        return " ".join(out.split())
    except Exception:
        return ""
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
def _strip_c_like_comments(text: str) -> str:
    """Strip // and /* */ comments while preserving strings and newlines."""
    out: list[str] = []
    i = 0
    n = len(text)
    state = "normal"
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "normal":
            if ch == "/" and nxt == "/":
                state = "line_comment"
                i += 2
                continue
            if ch == "/" and nxt == "*":
                state = "block_comment"
                i += 2
                continue
            out.append(ch)
            if ch == '"':
                state = "string"
            elif ch == "'":
                state = "char"
            i += 1
            continue
        if state == "line_comment":
            if ch in "\r\n":
                out.append(ch)
                state = "normal"
            i += 1
            continue
        if state == "block_comment":
            if ch in "\r\n":
                out.append(ch)
            if ch == "*" and nxt == "/":
                i += 2
                state = "normal"
            else:
                i += 1
            continue
        if state == "string":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == '"':
                state = "normal"
            i += 1
            continue
        if state == "char":
            out.append(ch)
            if ch == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 2
                continue
            if ch == "'":
                state = "normal"
            i += 1
            continue
    return "".join(out)
def _strip_python_comments(text: str) -> str:
    try:
        tokens = []
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.COMMENT:
                continue
            tokens.append(tok)
        return tokenize.untokenize(tokens)
    except tokenize.TokenError:
        return text
def _normalize_comment_stripped_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
def semantic_sha256_file(path: Path) -> str:
    raw = path.read_bytes()
    suffix = path.suffix.lower()
    if suffix in {".cu", ".cpp", ".cuh", ".h", ".hpp"}:
        normalized = _normalize_comment_stripped_text(_strip_c_like_comments(raw.decode("utf-8", errors="replace")))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    if suffix == ".py":
        normalized = _normalize_comment_stripped_text(_strip_python_comments(raw.decode("utf-8", errors="replace")))
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return hashlib.sha256(raw).hexdigest()
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
def resolve_arch_list(cfg: Dict[str, str] | None = None) -> str:
    cfg = cfg or read_config()
    native_only = _truthy(cfg.get("CUDA_NATIVE_GPU_ONLY"), default=True)
    keep_ptx = _truthy(cfg.get("CUDA_ARCH_KEEP_PTX"), default=False)
    arch_value = _norm_text(os.environ.get("TORCH_CUDA_ARCH_LIST") or cfg.get("TORCH_CUDA_ARCH_LIST") or "auto")
    fallback = _norm_text(cfg.get("CUDA_ARCH_FALLBACK")) or "8.6"
    needs_auto = (not arch_value) or arch_value.lower() in {"auto", "native", "gpu", "local"}
    if needs_auto:
        resolved = _detect_arch_with_nvidia_smi() or _detect_arch_with_torch() or fallback
    else:
        resolved = arch_value
    if not native_only:
        return resolved
    parts: list[str] = []
    for part in str(resolved).replace(",", ";").split(";"):
        item = _normalize_arch_part(part)
        if item and item not in parts:
            parts.append(item)
    if not parts:
        parts = [_normalize_arch_part(fallback)]
    out = parts[0]
    if keep_ptx:
        out += "+PTX"
    return out
def build_fingerprint(cfg_override: Dict[str, str] | None = None) -> Dict[str, object]:
    cfg = read_config()
    if cfg_override:
        for key, value in cfg_override.items():
            if key in BUILD_CONFIG_KEYS and value is not None:
                cfg[key] = str(value)
    files = []
    for path in SOURCE_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required build source not found: {path}")
        files.append({
            "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "semantic_sha256": semantic_sha256_file(path),
        })
    selected_cfg = {key: _norm_text(cfg.get(key)) for key in BUILD_CONFIG_KEYS}
    resolved_cuda_home = resolve_cuda_home(cfg)
    resolved_arch = resolve_arch_list(cfg)
    resolved = {
        "cuda_home": resolved_cuda_home,
        "nvcc_version": nvcc_version_text(cfg),
        "torch_cuda_arch_list": resolved_arch,
        "python_version": sys.version.split()[0],
        "python_executable_name": Path(sys.executable).name,
    }
    payload = {
        "schema": 8,
        "files": files,
        "build_config": selected_cfg,
        "resolved_build_inputs": resolved,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return {"digest": digest, "payload": payload}
def find_pyds() -> Tuple[List[Path], List[Path]]:
    local = sorted(CUDA_DIR.glob("avabm_cuda*.pyd"), key=lambda p: p.stat().st_mtime, reverse=True)
    root = sorted(ROOT_DIR.glob("avabm_cuda*.pyd"), key=lambda p: p.stat().st_mtime, reverse=True)
    return local, root
def newest(paths: Iterable[Path]) -> Path | None:
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)
def remove_pyds() -> None:
    for group in find_pyds():
        for path in group:
            try:
                path.unlink()
                print(f"[Info] Removed stale binary: {path}")
            except FileNotFoundError:
                pass
def copy_latest_pyd() -> Path | None:
    local, root = find_pyds()
    src = newest([*local, *root])
    if src is None:
        return None
    for dst_dir in (CUDA_DIR, ROOT_DIR):
        dst = dst_dir / src.name
        if dst.resolve() == src.resolve():
            continue
        if (not dst.exists()) or src.stat().st_mtime >= dst.stat().st_mtime:
            shutil.copy2(src, dst)
            print(f"[Info] Synced binary: {src.name} -> {dst_dir}")
    return src
def read_stamp() -> Dict[str, object] | None:
    try:
        return json.loads(STAMP_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
def is_up_to_date() -> bool:
    fp = build_fingerprint()
    stamp = read_stamp()
    if not stamp or stamp.get("digest") != fp["digest"]:
        return False
    local, root = find_pyds()
    if not local and not root:
        return False
    copy_latest_pyd()
    return True
def command_check(run_mode: bool = False) -> int:
    try:
        fp = build_fingerprint()
        if is_up_to_date():
            print("[Info] CUDA extension is up to date. Build is skipped.")
            print(f"[Info] Resolved arch: {fp['payload']['resolved_build_inputs']['torch_cuda_arch_list']}")
            return 0
        print("[Info] CUDA extension is not up to date. Build is required." if run_mode else "[Info] CUDA build required.")
        print(f"[Info] Resolved arch: {fp['payload']['resolved_build_inputs']['torch_cuda_arch_list']}")
        if run_mode:
            print("[Info] Runtime JIT build is disabled. Run avabm_cuda\\build.bat before run.bat.")
        return 2
    except Exception as exc:
        print(f"[Error] Build fingerprint check failed: {exc}")
        return 3
def command_detectarch() -> int:
    cfg = read_config()
    print(resolve_arch_list(cfg).replace("+PTX", ""))
    return 0
def _actual_build_env_from(env: Dict[str, str], cfg: Dict[str, str], phase: str) -> Dict[str, str]:
    actual = {key: _norm_text(cfg.get(key)) for key in BUILD_CONFIG_KEYS}
    for key in ACTUAL_ENV_OVERRIDE_KEYS:
        if env.get(key) is not None:
            actual[key] = _norm_text(env.get(key))
    actual["_build_phase"] = phase
    return actual
def _write_last_build_env(actual: Dict[str, str]) -> None:
    LAST_BUILD_ENV_PATH.write_text(json.dumps(actual, indent=2, sort_keys=True), encoding="utf-8")
def _read_last_build_env() -> Dict[str, str] | None:
    try:
        data = json.loads(LAST_BUILD_ENV_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return {str(k): _norm_text(v) for k, v in data.items()}
def _write_stream_header(f_last, f_dated, label: str, cmd: list[str], env: Dict[str, str]) -> None:
    header = (
        f"[Info] Build phase={label} started at unix={time.time():.3f}\n"
        f"[Info] cwd={CUDA_DIR}\n"
        f"[Info] command={' '.join(cmd)}\n"
        f"[Info] CUDA_HOME={env.get('CUDA_HOME', '')}\n"
        f"[Info] TORCH_CUDA_ARCH_LIST={env.get('TORCH_CUDA_ARCH_LIST', '')}\n"
        f"[Info] MAX_JOBS={env.get('MAX_JOBS', '')}\n"
        f"[Info] CUDA_PTXAS_SAFE_MODE={env.get('CUDA_PTXAS_SAFE_MODE', '')}\n"
        f"[Info] CUDA_NVCC_THREADS={env.get('CUDA_NVCC_THREADS', '')}\n"
        f"[Info] CUDA_SPLIT_COMPILE={env.get('CUDA_SPLIT_COMPILE', '')}\n"
    )
    f_last.write(header)
    f_dated.write(header)
    f_last.flush()
    f_dated.flush()
def _run_build_once(cmd: list[str], env: Dict[str, str], label: str, f_last, f_dated, verbose: bool) -> int:
    _write_stream_header(f_last, f_dated, label, cmd, env)
    proc = subprocess.Popen(
        cmd,
        cwd=str(CUDA_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert proc.stdout is not None
    start = time.monotonic()
    last_heartbeat = start
    for line in proc.stdout:
        f_last.write(line)
        f_dated.write(line)
        if verbose:
            print(line, end="", flush=True)
        now = time.monotonic()
        if (not verbose) and now - last_heartbeat >= 30.0:
            print(f"[Info] Build still running... phase={label} elapsed {int(now - start)}s")
            last_heartbeat = now
    rc = proc.wait()
    elapsed = time.monotonic() - start
    footer = f"[Info] Build phase={label} finished with exit code {rc} after {elapsed:.1f}s\n"
    f_last.write(footer)
    f_dated.write(footer)
    f_last.flush()
    f_dated.flush()
    return int(rc)
def _log_indicates_ptxas_crash(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return False
    low = text.lower()
    return (
        "ptxas" in low
        and (
            "access_violation" in low
            or "0xc0000005" in low
            or "died with status" in low
            or "internal error" in low
        )
    )
def _safe_retry_env(env: Dict[str, str]) -> Dict[str, str]:
    retry = env.copy()
    retry["CUDA_PTXAS_SAFE_MODE"] = "1"
    retry["CUDA_NVCC_THREADS"] = "0"
    retry["CUDA_SPLIT_COMPILE"] = "0"
    retry["CUDA_OPT_LEVEL"] = "0"
    retry["CUDA_PTXAS_OPT_LEVEL"] = "0"
    retry["CUDA_PTXAS_ALLOW_EXPENSIVE_OPT"] = "0"
    retry["CUDA_FAST_MATH"] = "0"
    retry["CUDA_BUILD_MODE"] = "fastdev"
    retry["CUDA_NVCC_ADD_STD_FLAG"] = "0"
    retry["CUDA_DEVICE_FORCEINLINE"] = "0"
    retry["CUDA_DEVICE_NOINLINE"] = "1"
    retry["MAX_JOBS"] = "1"
    retry["CMAKE_BUILD_PARALLEL_LEVEL"] = "1"
    retry["CUDA_BUILD_MAX_JOBS"] = "1"
    retry.pop("NVCC_PREPEND_FLAGS", None)
    retry.pop("NVCC_APPEND_FLAGS", None)
    return retry
def command_runbuild() -> int:
    """Run setup.py build_ext while always saving build logs."""
    cfg = read_config()
    log_dir = CUDA_DIR / "build_logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    last_log = log_dir / "build_last.log"
    dated_log = log_dir / f"build_{timestamp}.log"
    verbose = _truthy(os.environ.get("CUDA_VERBOSE_BUILD"), default=False)
    auto_retry = _truthy(os.environ.get("CUDA_AUTO_RETRY_PTXAS_CRASH") or cfg.get("CUDA_AUTO_RETRY_PTXAS_CRASH"), default=True)
    safe_retry_clean = _truthy(os.environ.get("CUDA_SAFE_RETRY_CLEAN") or cfg.get("CUDA_SAFE_RETRY_CLEAN"), default=True)
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    base_env = os.environ.copy()
    resolved_cuda_home = resolve_cuda_home(cfg)
    if resolved_cuda_home:
        base_env["CUDA_HOME"] = resolved_cuda_home
        base_env["PATH"] = str(Path(resolved_cuda_home) / "bin") + os.pathsep + base_env.get("PATH", "")
    base_env["TORCH_CUDA_ARCH_LIST"] = resolve_arch_list(cfg)
    if _truthy(cfg.get("CUDA_CLEAR_NVCC_GLOBAL_FLAGS"), default=True):
        base_env.pop("NVCC_PREPEND_FLAGS", None)
        base_env.pop("NVCC_APPEND_FLAGS", None)
    print(f"[Info] Running build command: {' '.join(cmd)}")
    print(f"[Info] Full build log: {last_log}")
    print(f"[Info] Timestamped log: {dated_log}")
    print(f"[Info] Resolved CUDA_HOME: {base_env.get('CUDA_HOME', '')}")
    print(f"[Info] Resolved TORCH_CUDA_ARCH_LIST: {base_env.get('TORCH_CUDA_ARCH_LIST', '')}")
    print(f"[Info] NVCC phase time CSV: {CUDA_DIR / 'build_logs' / 'nvcc_time.csv'}")
    if not verbose:
        print("[Info] CUDA_VERBOSE_BUILD=0. Compiler output is saved to log; only a summary is printed here.")
    with last_log.open("w", encoding="utf-8", errors="replace", newline="") as f_last, \
            dated_log.open("w", encoding="utf-8", errors="replace", newline="") as f_dated:
        actual_env = None
        rc = _run_build_once(cmd, base_env, "primary", f_last, f_dated, verbose)
        if rc == 0:
            actual_env = _actual_build_env_from(base_env, cfg, "primary")
        if rc != 0 and auto_retry and _log_indicates_ptxas_crash(last_log):
            print("[Warning] ptxas crash signature detected. Retrying once with serial safe ptxas settings.")
            f_last.write("[Warning] ptxas crash signature detected. Retrying once with serial safe ptxas settings.\n")
            f_dated.write("[Warning] ptxas crash signature detected. Retrying once with serial safe ptxas settings.\n")
            if safe_retry_clean:
                build_dir = CUDA_DIR / "build"
                if build_dir.exists():
                    shutil.rmtree(build_dir)
                    msg = f"[Info] Safe retry removed build cache: {build_dir}\n"
                    f_last.write(msg)
                    f_dated.write(msg)
                    print(msg.strip())
            retry_env = _safe_retry_env(base_env)
            rc = _run_build_once(cmd, retry_env, "safe_retry", f_last, f_dated, verbose)
            if rc == 0:
                actual_env = _actual_build_env_from(retry_env, cfg, "safe_retry")
    if rc == 0:
        if actual_env is not None:
            _write_last_build_env(actual_env)
            phase = actual_env.get("_build_phase", "primary")
            if phase != "primary":
                print("[Warning] Build succeeded via safe_retry; config.txt still requests the faster primary profile.")
                print("[Warning] The stamp will record the actual safe profile, so the next normal build will try the faster profile again.")
        print(f"[Info] Build command succeeded. Log saved: {last_log}")
    else:
        if LAST_BUILD_ENV_PATH.exists():
            LAST_BUILD_ENV_PATH.unlink()
        print(f"[Error] Build command failed with exit code {rc}. Log saved: {last_log}")
    return int(rc)
def command_prepare() -> int:
    try:
        remove_pyds()
        (CUDA_DIR / "build_logs").mkdir(exist_ok=True)
        return 0
    except Exception as exc:
        print(f"[Error] Failed to prepare CUDA build: {exc}")
        return 1
def command_clean() -> int:
    try:
        remove_pyds()
        if STAMP_PATH.exists():
            STAMP_PATH.unlink()
            print(f"[Info] Removed build stamp: {STAMP_PATH}")
        if LAST_BUILD_ENV_PATH.exists():
            LAST_BUILD_ENV_PATH.unlink()
            print(f"[Info] Removed last build env: {LAST_BUILD_ENV_PATH}")
        build_dir = CUDA_DIR / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print(f"[Info] Removed build cache: {build_dir}")
        return 0
    except Exception as exc:
        print(f"[Error] Failed to clean CUDA build: {exc}")
        return 1
def _add_cache_dir(candidates: list[tuple[Path, str]], value: str | None, reason: str) -> None:
    if not value:
        return
    expanded = os.path.expandvars(str(value).strip().strip('"'))
    if not expanded:
        return
    try:
        path = Path(expanded).expanduser()
    except OSError:
        return
    candidates.append((path, reason))
def _external_cache_candidates() -> list[tuple[Path, str]]:
    """Return conservative user-local CUDA/PyTorch cache directories.
    The list is intentionally restricted to well-known cache directory names.
    It never removes arbitrary parent directories just because an environment
    variable is set.
    """
    candidates: list[tuple[Path, str]] = []
    env = os.environ
    _add_cache_dir(candidates, env.get("TORCH_EXTENSIONS_DIR"), "TORCH_EXTENSIONS_DIR")
    for base_name in ("TEMP", "TMP", "LOCALAPPDATA"):
        base = env.get(base_name)
        if base:
            _add_cache_dir(candidates, str(Path(base) / "torch_extensions"), f"{base_name} torch_extensions")
    userprofile = env.get("USERPROFILE")
    if userprofile:
        _add_cache_dir(candidates, str(Path(userprofile) / "AppData" / "Local" / "Temp" / "torch_extensions"), "user Temp torch_extensions")
        _add_cache_dir(candidates, str(Path(userprofile) / ".cache" / "torch_extensions"), "user .cache torch_extensions")
    _add_cache_dir(candidates, env.get("CUDA_CACHE_PATH"), "CUDA_CACHE_PATH")
    for base_name in ("APPDATA", "LOCALAPPDATA"):
        base = env.get(base_name)
        if base:
            _add_cache_dir(candidates, str(Path(base) / "NVIDIA" / "ComputeCache"), f"{base_name} NVIDIA ComputeCache")
    if userprofile:
        _add_cache_dir(candidates, str(Path(userprofile) / ".nv" / "ComputeCache"), "user .nv ComputeCache")
    _add_cache_dir(candidates, str(ROOT_DIR / "__pycache__"), "project Python bytecode")
    _add_cache_dir(candidates, str(CUDA_DIR / "__pycache__"), "CUDA helper Python bytecode")
    _add_cache_dir(candidates, str(CUDA_DIR / "avabm_cuda.egg-info"), "setuptools egg-info")
    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for path, reason in candidates:
        try:
            key = os.path.normcase(os.path.abspath(path))
        except OSError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique.append((Path(key) if os.name != "nt" else path, reason))
    return unique
def _is_safe_cache_dir(path: Path) -> bool:
    """Guard against deleting broad directories from mis-set env vars."""
    try:
        name = path.name.lower()
        full = str(path).lower()
    except OSError:
        return False
    if name in {"torch_extensions", "computecache", "__pycache__", "avabm_cuda.egg-info"}:
        return True
    if "torch_extensions" in full and "temp" in full:
        return True
    if "nvidia" in full and "computecache" in full:
        return True
    return False
def _remove_cache_dir(path: Path, reason: str) -> None:
    if not path.exists():
        return
    if not path.is_dir():
        print(f"[Info] Skipped non-directory cache candidate: {path}")
        return
    if not _is_safe_cache_dir(path):
        print(f"[Warning] Skipped unsafe-looking cache path from {reason}: {path}")
        return
    shutil.rmtree(path)
    print(f"[Info] Removed external cache ({reason}): {path}")
def command_hardclean() -> int:
    """Clean project outputs plus conservative user-local CUDA/PyTorch caches."""
    try:
        rc = command_clean()
        if rc != 0:
            return rc
        for path, reason in _external_cache_candidates():
            try:
                _remove_cache_dir(path, reason)
            except PermissionError as exc:
                print(f"[Warning] Permission denied while removing cache {path}: {exc}")
            except OSError as exc:
                print(f"[Warning] Could not remove cache {path}: {exc}")
        print("[Info] Hard clean completed. Next build will be forced and no runtime JIT build is used.")
        return 0
    except Exception as exc:
        print(f"[Error] Failed to hard-clean CUDA/PyTorch caches: {exc}")
        return 1
def command_mark() -> int:
    try:
        pyd = copy_latest_pyd()
        if pyd is None:
            print("[Error] Build finished, but no avabm_cuda*.pyd was found.")
            return 1
        actual_env = _read_last_build_env()
        phase = actual_env.get("_build_phase", "primary") if actual_env else "primary"
        fp = build_fingerprint()
        stamp = {
            "digest": fp["digest"],
            "payload": fp["payload"],
            "actual_build_phase": phase,
            "actual_build_env": actual_env or {},
            "binary": pyd.name,
            "python": sys.version.split()[0],
            "updated_at_unix": time.time(),
        }
        STAMP_PATH.write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Info] Updated CUDA build stamp: {STAMP_PATH}")
        return 0
    except Exception as exc:
        print(f"[Error] Failed to finalize CUDA build: {exc}")
        return 1
def command_status() -> int:
    fp = build_fingerprint()
    local, root = find_pyds()
    print(f"[Info] Fingerprint: {fp['digest']}")
    print(f"[Info] Resolved inputs: {json.dumps(fp['payload']['resolved_build_inputs'], ensure_ascii=False)}")
    print(f"[Info] Local pyds: {[p.name for p in local]}")
    print(f"[Info] Root pyds: {[p.name for p in root]}")
    print(f"[Info] Stamp exists: {STAMP_PATH.exists()}")
    return 0
def main(argv: List[str]) -> int:
    cmd = argv[1].lower() if len(argv) > 1 else "check"
    if cmd == "check":
        return command_check(run_mode=False)
    if cmd in {"runcheck", "verify-run"}:
        return command_check(run_mode=True)
    if cmd == "detectarch":
        return command_detectarch()
    if cmd == "runbuild":
        return command_runbuild()
    if cmd == "prepare":
        return command_prepare()
    if cmd == "clean":
        return command_clean()
    if cmd in {"hardclean", "cleanall", "clean-all"}:
        return command_hardclean()
    if cmd == "mark":
        return command_mark()
    if cmd == "status":
        return command_status()
    print(f"[Error] Unknown build helper command: {cmd}")
    return 64
if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
