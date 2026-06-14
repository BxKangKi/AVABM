"""Small build helper for the AVABM CUDA extension.

This file keeps build.bat simple and makes Windows CUDA builds much faster in
normal edit/run cycles:

* It computes a fingerprint from the CUDA/C++ source files and only the
  build-related config.txt keys.  Runtime traffic parameters do not force a
  CUDA rebuild.  Comment-only code edits still change the source hash, so the
  first run after this annotated patch must rebuild once; later runs are skipped.
* It skips build_ext entirely when the compiled .pyd already matches the
  fingerprint.
* When a rebuild is required it removes only stale .pyd outputs by default,
  leaving the Ninja object cache intact.  A full clean is still available with
  `build.bat clean` or CUDA_FORCE_REBUILD=1.

Korean summary:
* main.cu/binding.cpp/setup.py/pyproject.toml 및 빌드 관련 config만 해시화합니다.
* 시뮬레이션 파라미터만 바꾼 경우 CUDA를 다시 컴파일하지 않습니다.
* 일반 빌드는 build 폴더를 지우지 않아 Ninja 증분 빌드 캐시를 재사용합니다.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

CUDA_DIR = Path(__file__).resolve().parent
ROOT_DIR = CUDA_DIR.parent
STAMP_PATH = CUDA_DIR / ".avabm_cuda_build_fingerprint.json"

SOURCE_FILES = [
    CUDA_DIR / "main.cu",
    CUDA_DIR / "binding.cpp",
    CUDA_DIR / "setup.py",
    CUDA_DIR / "pyproject.toml",
]

# Only these config keys affect CUDA compilation.  Ordinary simulation settings
# should not make users wait for another nvcc build.
# KO 문법 이유: list 안에 키 이름을 문자열로 모아 두면 fingerprint 계산에서
#    동일한 순서로 config 값을 읽을 수 있습니다.
# KO 논리 이유: CUDA_BUILD_MODE/CUDA_NVCC_THREADS 같은 빌드 속도 옵션은 바이너리
#    생성 명령을 바꾸므로 해시에 포함하고, BASE_VPS 같은 실행 파라미터는 제외합니다.
BUILD_CONFIG_KEYS = [
    "CUDA_HOME",
    "TORCH_CUDA_ARCH_LIST",
    "TORCH_NVCC_FLAGS",
    "TORCH_DONT_CHECK_COMPILER_ABI",
    "CUDA_EXTRA_NVCC_FLAGS",
    "CUDA_EXTRA_CXX_FLAGS",
    "CUDA_SUPPRESS_HEADER_WARNINGS",
    "CUDA_SHOW_WARNINGS",
    "CUDA_BUILD_MODE",
    "CUDA_CXX_STANDARD",
    "CUDA_DISABLE_MSVC_GL",
    "CUDA_USE_FULL_TORCH_EXTENSION_HEADER",
    "CUDA_OPT_LEVEL",
    "CUDA_FAST_MATH",
    "CUDA_FAST_EQUIV_MATH",
    "CUDA_USE_ASYNC_MEMSET_CLEAR",
    "CUDA_SPAWN_GRID_INSERT_FASTPATH",
    "SPEED_MIN_CRUISE_ENABLED",
    "SPEED_MIN_CRUISE_KMH",
    "CUDA_NVCC_THREADS",
    "CUDA_SPLIT_COMPILE",
]


def _norm_text(value: str | None) -> str:
    return "" if value is None else str(value).strip()


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_fingerprint() -> Dict[str, object]:
    cfg = read_config()
    files = []
    for path in SOURCE_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Required build source not found: {path}")
        files.append({
            "path": str(path.relative_to(ROOT_DIR)).replace("\\", "/"),
            "sha256": sha256_file(path),
        })
    selected_cfg = {key: _norm_text(cfg.get(key)) for key in BUILD_CONFIG_KEYS}
    payload = {
        "schema": 2,
        "files": files,
        "build_config": selected_cfg,
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

    # Keep a copy beside build.bat and a copy in the project root.  main.py imports
    # the root copy, while build_ext normally creates the local copy.
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
        if is_up_to_date():
            print("[Info] CUDA extension is up to date. Build is skipped.")
            return 0
        print("[Info] CUDA extension is not up to date. Build is required." if run_mode else "[Info] CUDA build required.")
        return 2
    except Exception as exc:  # noqa: BLE001 - printed for batch users
        print(f"[Error] Build fingerprint check failed: {exc}")
        return 3


def command_prepare() -> int:
    try:
        # Delete only old .pyd outputs so a failed rebuild cannot accidentally run
        # an old binary.  Do NOT delete build/ here; Ninja can reuse .obj/.d files.
        remove_pyds()
        (CUDA_DIR / "build_logs").mkdir(exist_ok=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[Error] Failed to prepare CUDA build: {exc}")
        return 1


def command_clean() -> int:
    try:
        remove_pyds()
        if STAMP_PATH.exists():
            STAMP_PATH.unlink()
            print(f"[Info] Removed build stamp: {STAMP_PATH}")
        build_dir = CUDA_DIR / "build"
        if build_dir.exists():
            shutil.rmtree(build_dir)
            print(f"[Info] Removed build cache: {build_dir}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[Error] Failed to clean CUDA build: {exc}")
        return 1


def command_mark() -> int:
    try:
        pyd = copy_latest_pyd()
        if pyd is None:
            print("[Error] Build finished, but no avabm_cuda*.pyd was found.")
            return 1
        fp = build_fingerprint()
        stamp = {
            "digest": fp["digest"],
            "payload": fp["payload"],
            "binary": pyd.name,
            "python": sys.version.split()[0],
            "updated_at_unix": time.time(),
        }
        STAMP_PATH.write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[Info] Updated CUDA build stamp: {STAMP_PATH}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[Error] Failed to finalize CUDA build: {exc}")
        return 1


def command_status() -> int:
    fp = build_fingerprint()
    local, root = find_pyds()
    print(f"[Info] Fingerprint: {fp['digest']}")
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
    if cmd == "prepare":
        return command_prepare()
    if cmd == "clean":
        return command_clean()
    if cmd == "mark":
        return command_mark()
    if cmd == "status":
        return command_status()
    print(f"[Error] Unknown build helper command: {cmd}")
    return 64


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
