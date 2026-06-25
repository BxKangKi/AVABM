#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT_DIR="$(pwd)"
CUDA_DIR="$ROOT_DIR/avabm_cuda"
CPU_DIR="$ROOT_DIR/avabm_cpu"
CONFIG_FILE="$ROOT_DIR/config.txt"
BUILD_HELPER="$CUDA_DIR/build_helper.py"

load_config() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r raw || [ -n "$raw" ]; do
    case "$raw" in
      ''|'#'*|';'*) continue ;;
    esac
    case "$raw" in *=*) ;; *) continue ;; esac
    local key val
    key="${raw%%=*}"
    val="${raw#*=}"
    key="$(printf '%s' "$key" | sed 's/[[:space:]]//g')"
    case "$key" in ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;; esac
    val="$(printf '%s' "$val" | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*;.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//')"
    export "$key=$val"
  done < "$file"
}

load_config "$CONFIG_FILE"

: "${CONDA_ENV:=avabm}"
: "${PYTHON_COMMAND:=python}"
: "${CUDA_BUILD_LOG_TAIL:=200}"
: "${CUDA_FORCE_REBUILD:=0}"
: "${CUDA_SKIP_IF_UP_TO_DATE:=1}"
: "${CUDA_INCREMENTAL_BUILD:=1}"
: "${MAX_JOBS:=1}"
: "${CUDA_BUILD_MAX_JOBS:=1}"
: "${SIM_BACKEND:=auto}"
: "${CPU_WORKERS:=0}"
: "${CPU_AUTO_BUILD:=1}"
: "${CPU_BUILD_MAX_JOBS:=0}"

if [ ! -f "$BUILD_HELPER" ]; then
  echo "[Error] Missing CUDA build helper: $BUILD_HELPER" >&2
  exit 1
fi

activate_conda() {
  if [ "${AVABM_CONDA_READY:-0}" = "1" ]; then
    return 0
  fi

  if command -v conda >/dev/null 2>&1; then
    local conda_setup
    conda_setup="$(conda shell.bash hook 2>/dev/null || true)"
    if [ -n "$conda_setup" ]; then
      eval "$conda_setup"
    fi
  elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/miniforge3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/miniconda3/etc/profile.d/conda.sh"
  elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/anaconda3/etc/profile.d/conda.sh"
  fi

  if command -v conda >/dev/null 2>&1; then
    if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
      echo "[Info] Creating Conda environment: $CONDA_ENV"
      conda create -n "$CONDA_ENV" python=3.12 ipykernel -y
    fi
    echo "[Info] Activating Conda environment: $CONDA_ENV"
    conda activate "$CONDA_ENV"
  else
    echo "[Warning] conda was not found. Continuing with the current shell Python."
  fi

  export AVABM_CONDA_READY=1
}

cap_max_jobs() {
  local mj cap
  mj="$MAX_JOBS"
  cap="$CUDA_BUILD_MAX_JOBS"
  case "$mj" in ''|*[!0-9]*) mj=1 ;; esac
  case "$cap" in ''|*[!0-9]*) cap=1 ;; esac
  if [ "$mj" -le 0 ]; then mj=1; fi
  if [ "$cap" -le 0 ]; then cap=1; fi
  if [ "$mj" -gt "$cap" ]; then mj="$cap"; fi
  export MAX_JOBS="$mj"
  export CMAKE_BUILD_PARALLEL_LEVEL="$mj"
}

ensure_ninja() {
  if command -v ninja >/dev/null 2>&1; then
    return 0
  fi
  if "$PYTHON_COMMAND" -c 'import ninja' >/dev/null 2>&1; then
    return 0
  fi
  if command -v conda >/dev/null 2>&1; then
    echo "[Info] ninja not found. Installing ninja into the Conda environment..."
    conda install -n "$CONDA_ENV" ninja -y
  else
    echo "[Warning] ninja was not found. Install ninja if the PyTorch build backend asks for it."
  fi
}

build_cuda() {
  local request rc log
  request="${1:-auto}"
  activate_conda
  cap_max_jobs
  ensure_ninja
  export USE_NINJA=1

  case "$request" in
    hardclean|cleanall|clean-all)
      echo "[Info] Hard clean rebuild requested."
      "$PYTHON_COMMAND" "$BUILD_HELPER" hardclean
      ;;
    clean|rebuild)
      echo "[Info] Full clean rebuild requested."
      "$PYTHON_COMMAND" "$BUILD_HELPER" clean
      ;;
    *)
      if [ "$CUDA_FORCE_REBUILD" = "1" ]; then
        echo "[Info] CUDA_FORCE_REBUILD=1; cleaning before build."
        "$PYTHON_COMMAND" "$BUILD_HELPER" clean
      elif [ "$CUDA_SKIP_IF_UP_TO_DATE" = "1" ]; then
        set +e
        "$PYTHON_COMMAND" "$BUILD_HELPER" check
        rc=$?
        set -e
        case "$rc" in
          0)
            echo "[Info] CUDA build skipped: compiled module already matches the source fingerprint."
            return 0
            ;;
          2)
            ;;
          *)
            echo "[Error] Build status check failed."
            return 1
            ;;
        esac
      fi
      ;;
  esac

  if [ "$request" != "clean" ] && [ "$request" != "rebuild" ] && [ "$request" != "hardclean" ] && [ "$request" != "cleanall" ] && [ "$request" != "clean-all" ]; then
    if [ "$CUDA_INCREMENTAL_BUILD" = "1" ] && [ "$CUDA_FORCE_REBUILD" != "1" ]; then
      echo "[Info] Preparing incremental build: remove stale extension binary only, keep build cache."
      "$PYTHON_COMMAND" "$BUILD_HELPER" prepare
    elif [ "$CUDA_FORCE_REBUILD" != "1" ]; then
      echo "[Info] Incremental build disabled, cleaning full build cache."
      "$PYTHON_COMMAND" "$BUILD_HELPER" clean
    fi
  fi

  mkdir -p "$CUDA_DIR/build_logs"
  log="$CUDA_DIR/build_logs/build_last.log"
  echo "[Info] Building CUDA extension in place..."
  echo "[Info] Full build log: $log"
  set +e
  "$PYTHON_COMMAND" "$BUILD_HELPER" runbuild
  rc=$?
  set -e
  if [ "$rc" -ne 0 ]; then
    echo "======================================================="
    echo "[Error] CUDA build failed. No old extension binary will be used."
    echo "[Error] The last $CUDA_BUILD_LOG_TAIL log lines are printed below."
    echo "[Error] Full log: $log"
    echo "======================================================="
    if [ -f "$log" ]; then
      tail -n "$CUDA_BUILD_LOG_TAIL" "$log" || true
    fi
    return 1
  fi

  echo "[Info] Build succeeded. Short log tail:"
  if [ -f "$log" ]; then
    tail -n 40 "$log" || true
  fi
  "$PYTHON_COMMAND" "$BUILD_HELPER" mark
  echo "======================================================="
  echo "[Info] CUDA build completed successfully."
  echo "======================================================="
}

ensure_cuda_built() {
  local rc
  activate_conda
  set +e
  "$PYTHON_COMMAND" "$BUILD_HELPER" verify-run
  rc=$?
  set -e
  case "$rc" in
    0) return 0 ;;
    2)
      echo "[Info] CUDA extension is missing or stale. Building now..."
      build_cuda auto
      ;;
    *)
      echo "[Error] CUDA build fingerprint check failed."
      return 1
      ;;
  esac
}

python_cuda_available() {
  activate_conda
  "$PYTHON_COMMAND" - <<'PY' >/dev/null 2>&1
import sys
try:
    import torch
    sys.exit(0 if torch.cuda.is_available() else 1)
except Exception:
    sys.exit(1)
PY
}

build_cpu() {
  activate_conda
  if [ ! -f "$CPU_DIR/setup.py" ]; then
    echo "[Error] Missing CPU backend setup.py: $CPU_DIR/setup.py" >&2
    return 1
  fi
  echo "[Info] Building C++ CPU extension in place..."
  (cd "$CPU_DIR" && "$PYTHON_COMMAND" setup.py build_ext --inplace)
}

ensure_cpu_built() {
  local rc
  activate_conda
  set +e
  "$PYTHON_COMMAND" -c 'import avabm_cpu' >/dev/null 2>&1
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  if [ "${CPU_AUTO_BUILD:-1}" != "1" ]; then
    echo "[Error] CPU backend is missing and CPU_AUTO_BUILD is disabled." >&2
    echo "[Error] Build it with: (cd avabm_cpu && $PYTHON_COMMAND setup.py build_ext --inplace)" >&2
    return 1
  fi
  build_cpu
}

backend_from_args() {
  local requested arg
  requested="${SIM_BACKEND:-auto}"
  for arg in "$@"; do
    case "$arg" in
      --cpu|--backend=cpu) requested="cpu" ;;
      --gpu|--cuda|--backend=cuda) requested="cuda" ;;
      --backend=cuda_strict|--cuda-strict) requested="cuda_strict" ;;
    esac
  done
  printf '%s' "$requested" | tr '[:upper:]' '[:lower:]'
}

ensure_backend_built() {
  local requested rc
  requested="$(backend_from_args "$@")"
  case "$requested" in
    cpu|cpp_cpu|c++)
      ensure_cpu_built
      ;;
    cuda_strict)
      ensure_cuda_built
      ;;
    cuda|gpu)
      if python_cuda_available; then
        set +e
        ensure_cuda_built
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then
          return 0
        fi
        echo "[Warning] CUDA backend build/verification failed. Falling back to CPU backend."
        ensure_cpu_built
      else
        echo "[Warning] SIM_BACKEND=$requested requested, but torch.cuda.is_available() is false. Falling back to CPU backend."
        ensure_cpu_built
      fi
      ;;
    auto|*)
      if python_cuda_available; then
        set +e
        ensure_cuda_built
        rc=$?
        set -e
        if [ "$rc" -eq 0 ]; then
          return 0
        fi
        echo "[Warning] CUDA backend build/verification failed. Falling back to CPU backend."
      else
        echo "[Info] CUDA is unavailable. Using C++ CPU backend."
      fi
      ensure_cpu_built
      ;;
  esac
}

launch_mode() {
  local mode
  mode="$1"
  shift || true
  activate_conda
  export SIM_ENGINE=ecs_abm_cuda
  export SIMULATION_ENGINE=ecs_abm_cuda
  case "$mode" in
    turbo)
      export HEADLESS_MODE=1
      export SIM_HEADLESS=1
      export ABM_TURBO_PROFILE=1
      echo "[Info] Starting AVABM in Turbo mode."
      exec "$PYTHON_COMMAND" main.py --ecs --turbo "$@"
      ;;
    visual)
      export HEADLESS_MODE=0
      export SIM_HEADLESS=0
      echo "[Info] Starting AVABM in Visual mode."
      exec "$PYTHON_COMMAND" main.py --ecs --visual "$@"
      ;;
    *)
      echo "[Error] Unknown run mode: $mode" >&2
      return 1
      ;;
  esac
}

show_menu() {
  echo
  echo "======================================================="
  echo "AVABM Launcher"
  echo "======================================================="
  echo "1. Turbo   - headless selected-backend batch run"
  echo "2. Visual  - OpenGL/Pygame selected-backend window run"
  echo "3. Build selected backend from SIM_BACKEND"
  echo "4. Build CUDA only (skip if up to date)"
  echo "5. Build CPU only"
  echo "6. Clean rebuild CUDA"
  echo "7. Hard clean + rebuild CUDA"
  echo "8. CUDA build status"
  echo "9. Exit"
  echo
  printf 'Select [1-9]: '
}

cmd="${1:-}"
if [ -z "$cmd" ]; then
  show_menu
  read -r choice
  case "$choice" in
    1|t|T|turbo|Turbo) set -- turbo ;;
    2|v|V|visual|Visual) set -- visual ;;
    3|b|B|build|Build) set -- build ;;
    4|bc|build-cuda|cuda-build) set -- build-cuda ;;
    5|cpu|build-cpu|cpu-build) set -- build-cpu ;;
    6|r|R|rebuild|Rebuild|clean|Clean) set -- rebuild ;;
    7|h|H|hardclean|Hardclean|cleanall|clean-all) set -- hardclean ;;
    8|s|S|status|Status|check|Check) set -- status ;;
    9|q|Q|exit|Exit) exit 0 ;;
    *) echo "[Error] Invalid selection." >&2; exit 1 ;;
  esac
  cmd="$1"
  shift || true
else
  shift || true
fi

case "$cmd" in
  turbo|--turbo|headless|--headless)
    ensure_backend_built "$@"
    launch_mode turbo "$@"
    ;;
  visual|--visual|gui|--gui|window|--window)
    ensure_backend_built "$@"
    launch_mode visual "$@"
    ;;
  build|build-selected|selected-build)
    ensure_backend_built "$@"
    ;;
  build-cuda|cuda-build)
    build_cuda auto
    ;;
  build-cpu|cpu-build)
    build_cpu
    ;;
  rebuild|clean)
    build_cuda clean
    ;;
  hardclean|cleanall|clean-all)
    build_cuda hardclean
    ;;
  status|check)
    activate_conda
    "$PYTHON_COMMAND" "$BUILD_HELPER" status
    ;;
  *)
    echo "[Error] Unknown command: $cmd" >&2
    echo "Usage: ./run.sh [turbo|visual|build|build-cuda|build-cpu|rebuild|hardclean|status] [extra main.py args...]" >&2
    exit 1
    ;;
esac
