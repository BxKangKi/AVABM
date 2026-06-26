#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT_DIR="$(pwd)"
PACKAGE_DIR="$ROOT_DIR/avabm"
CUDA_DIR="$PACKAGE_DIR/cuda"
CPU_DIR="$PACKAGE_DIR/cpu"
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
  echo "[Info] Building avabm CPU extension in place..."
  (cd "$CPU_DIR" && "$PYTHON_COMMAND" setup.py build_ext --inplace)
}

build_all() {
  build_cpu
  build_cuda auto
}

ensure_cpu_built() {
  local rc
  activate_conda
  set +e
  "$PYTHON_COMMAND" - <<'PY' >/dev/null 2>&1
import avabm
avabm.import_cpu()
PY
  rc=$?
  set -e
  if [ "$rc" -eq 0 ]; then
    return 0
  fi
  if [ "${CPU_AUTO_BUILD:-1}" != "1" ]; then
    echo "[Error] CPU backend is missing and CPU_AUTO_BUILD is disabled." >&2
    echo "[Error] Build it with: (cd avabm/cpu && $PYTHON_COMMAND setup.py build_ext --inplace)" >&2
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

benchmark_order_from_args() {
  local arg value order low
  order=""
  for arg in "$@"; do
    value=""
    case "$arg" in
      cpu|CPU|benchmark-cpu|bench-cpu|--cpu|--benchmark-cpu|--bench-cpu|--backend=cpu|--benchmark-backend=cpu|--benchmark-backends=cpu|--benchmark-order=cpu)
        value="cpu" ;;
      gpu|GPU|cuda|CUDA|benchmark-gpu|benchmark-cuda|bench-gpu|bench-cuda|--gpu|--cuda|--cuda-strict|--benchmark-gpu|--benchmark-cuda|--bench-gpu|--bench-cuda|--backend=cuda|--backend=cuda_strict|--benchmark-backend=gpu|--benchmark-backend=cuda|--benchmark-backends=gpu|--benchmark-backends=cuda|--benchmark-order=gpu|--benchmark-order=cuda)
        value="cuda" ;;
      sumo|SUMO|benchmark-sumo|bench-sumo|--sumo|--benchmark-sumo|--bench-sumo|baseline|sumo-baseline|--benchmark-backend=sumo|--benchmark-backends=sumo|--benchmark-order=sumo)
        value="sumo" ;;
      both|Both|cpu,cuda|cuda,cpu|--benchmark-order=cpu,cuda|--benchmark-order=cuda,cpu|--benchmark-backends=cpu,cuda|--benchmark-backends=cuda,cpu)
        value="cpu,cuda" ;;
      all|All|compare|Compare|cpu,cuda,sumo|cuda,cpu,sumo|--benchmark-order=cpu,cuda,sumo|--benchmark-order=cuda,cpu,sumo|--benchmark-backends=cpu,cuda,sumo|--benchmark-backends=cuda,cpu,sumo)
        value="cpu,cuda,sumo" ;;
      gpu+sumo|cuda+sumo|gpu-sumo|cuda-sumo|sumo+gpu|sumo+cuda|sumo-gpu|sumo-cuda|--benchmark-order=cuda,sumo|--benchmark-order=gpu,sumo|--benchmark-backends=cuda,sumo|--benchmark-backends=gpu,sumo)
        value="cuda,sumo" ;;
      cpu+sumo|cpu-sumo|sumo+cpu|sumo-cpu|--benchmark-order=cpu,sumo|--benchmark-backends=cpu,sumo)
        value="cpu,sumo" ;;
      --benchmark-order=*|--benchmark-backend=*|--benchmark-backends=*)
        value="${arg#*=}" ;;
      --with-sumo|--benchmark-with-sumo|--include-sumo)
        if [ -n "$order" ]; then value="$order,sumo"; else value="cpu,cuda,sumo"; fi ;;
    esac
    low="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
    case "$low" in
      cpu|cpp|cpp_cpu|c++) order="cpu" ;;
      gpu|cuda|cuda_strict|cuda-strict) order="cuda" ;;
      sumo|sumo_cli|sumo-cli|baseline|sumo-baseline) order="sumo" ;;
      both|cpu,cuda|cuda,cpu) order="cpu,cuda" ;;
      all|compare|cpu,cuda,sumo|cuda,cpu,sumo|sumo,cpu,cuda|sumo,cuda,cpu) order="cpu,cuda,sumo" ;;
      gpu+sumo|cuda+sumo|gpu-sumo|cuda-sumo|sumo+gpu|sumo+cuda|sumo-gpu|sumo-cuda|cuda,sumo|gpu,sumo|sumo,cuda|sumo,gpu) order="cuda,sumo" ;;
      cpu+sumo|cpu-sumo|sumo+cpu|sumo-cpu|cpu,sumo|sumo,cpu) order="cpu,sumo" ;;
    esac
  done
  printf '%s' "$order"
}

ensure_benchmark_backends_built() {
  local order need_cpu need_cuda
  order="$(benchmark_order_from_args "$@")"
  if [ -z "$order" ]; then
    order="cpu,cuda"
  fi
  need_cpu=0
  need_cuda=0
  case ",$order," in *",cpu,"*) need_cpu=1 ;; esac
  case ",$order," in *",cuda,"*) need_cuda=1 ;; esac
  if [ "$need_cpu" = "1" ]; then
    ensure_cpu_built
  fi
  if [ "$need_cuda" = "1" ]; then
    ensure_cuda_built
  fi
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
    benchmark)
      export HEADLESS_MODE=1
      export SIM_HEADLESS=1
      export BENCHMARK_MODE=1
      export ABM_TURBO_PROFILE=1
      echo "[Info] Starting AVABM benchmark mode (use: benchmark cpu | gpu | sumo | both | all; add --with-sumo; load: --benchmark-spawn-vps=100)."
      exec "$PYTHON_COMMAND" main.py --benchmark "$@"
      ;;
    *)
      echo "[Error] Unknown run mode: $mode" >&2
      return 1
      ;;
  esac
}


normalize_benchmark_order() {
  local raw token low out canon
  raw="${1:-}"
  raw="${raw// /,}"
  raw="${raw//\"/}"
  raw="${raw//\'/}"
  out=""
  IFS=',' read -r -a _avabm_order_tokens <<< "$raw"
  for token in "${_avabm_order_tokens[@]}"; do
    low="${token,,}"
    low="${low//[[:space:]]/}"
    [ -n "$low" ] || continue
    case "$low" in
      cpu|cpp|cpp_cpu|c++)
        canon="cpu"
        case ",$out," in *",$canon,"*) ;; *) out="${out:+$out,}$canon" ;; esac
        ;;
      gpu|cuda|cuda_strict|cuda-strict)
        canon="cuda"
        case ",$out," in *",$canon,"*) ;; *) out="${out:+$out,}$canon" ;; esac
        ;;
      sumo|sumo_cli|sumo-cli|baseline|sumo-baseline)
        canon="sumo"
        case ",$out," in *",$canon,"*) ;; *) out="${out:+$out,}$canon" ;; esac
        ;;
      both)
        for canon in cpu cuda; do
          case ",$out," in *",$canon,"*) ;; *) out="${out:+$out,}$canon" ;; esac
        done
        ;;
      all|compare)
        for canon in cpu cuda sumo; do
          case ",$out," in *",$canon,"*) ;; *) out="${out:+$out,}$canon" ;; esac
        done
        ;;
    esac
  done
  printf '%s' "$out"
}

benchmark_selector() {
  local choice order low load value steps warmup max_agents sumo_runner
  BENCHMARK_MENU_ORDER=""
  BENCHMARK_MENU_ARGS=()

  while true; do
    if [ -t 1 ]; then clear || true; fi
    echo
    echo "======================================================="
    echo "AVABM Benchmark Architecture Selector"
    echo "======================================================="
    echo "Choose which architecture/backend to benchmark."
    echo
    echo "1. CPU only                 (C++ CPU backend)"
    echo "2. GPU only                 (CUDA backend)"
    echo "3. SUMO only                (SUMO baseline)"
    echo "4. CPU + GPU                (AVABM internal comparison)"
    echo "5. GPU + SUMO               (recommended for SUMO baseline)"
    echo "6. CPU + SUMO"
    echo "7. CPU + GPU + SUMO         (all three)"
    echo "8. Custom order/subset      (example: cuda,sumo)"
    echo "0. Cancel"
    echo
    read -r -p "Select [1-8,0]: " choice
    choice="${choice// /}"
    choice="${choice//\"/}"
    low="${choice,,}"
    case "$low" in
      0|q|quit|exit) return 2 ;;
      1|cpu) order="cpu" ;;
      2|gpu|cuda) order="cuda" ;;
      3|sumo) order="sumo" ;;
      4|both|cpu+gpu|cpu-gpu|cpu+cuda|cpu-cuda) order="cpu,cuda" ;;
      5|gpu+sumo|cuda+sumo|gpu-sumo|cuda-sumo|sumo+gpu|sumo+cuda|sumo-gpu|sumo-cuda) order="cuda,sumo" ;;
      6|cpu+sumo|cpu-sumo|sumo+cpu|sumo-cpu) order="cpu,sumo" ;;
      7|all|compare) order="cpu,cuda,sumo" ;;
      8|custom)
        echo
        echo "Enter comma-separated backends in run order."
        echo "Valid names: cpu, cuda, gpu, sumo. Example: cuda,sumo"
        read -r -p "Order: " value
        order="$(normalize_benchmark_order "$value")"
        if [ -z "$order" ]; then
          echo "[Warning] No valid benchmark backend was entered."
          read -r -p "Press Enter to retry..." _
          continue
        fi
        ;;
      *)
        echo "[Warning] Invalid benchmark architecture selection: $choice"
        read -r -p "Press Enter to retry..." _
        continue
        ;;
    esac
    BENCHMARK_MENU_ORDER="$order"
    break
  done

  while true; do
    echo
    echo "-------------------------------------------------------"
    echo "Benchmark load / spawn setting"
    echo "-------------------------------------------------------"
    echo "1. Use config/default demand"
    echo "2. 20 vehicles/sec total spawn target"
    echo "3. 50 vehicles/sec total spawn target"
    echo "4. 100 vehicles/sec total spawn target"
    echo "5. 200 vehicles/sec total spawn target"
    echo "6. Custom vehicles/sec total spawn target"
    echo "7. Custom total spawned vehicles during timed benchmark"
    echo "8. Custom vehicles/sec per spawn point"
    echo "0. Cancel"
    echo
    read -r -p "Select [1-8,0, Enter=1]: " load
    load="${load// /}"
    load="${load//\"/}"
    [ -n "$load" ] || load="1"
    case "${load,,}" in
      0|q|quit|exit) return 2 ;;
      1) ;;
      2) BENCHMARK_MENU_ARGS+=("--benchmark-spawn-vps=20") ;;
      3) BENCHMARK_MENU_ARGS+=("--benchmark-spawn-vps=50") ;;
      4) BENCHMARK_MENU_ARGS+=("--benchmark-spawn-vps=100") ;;
      5) BENCHMARK_MENU_ARGS+=("--benchmark-spawn-vps=200") ;;
      6)
        read -r -p "Total spawn VPS: " value
        value="${value// /}"
        value="${value//\"/}"
        [ -n "$value" ] || continue
        BENCHMARK_MENU_ARGS+=("--benchmark-spawn-vps=$value")
        ;;
      7)
        read -r -p "Total spawned vehicles during timed benchmark: " value
        value="${value// /}"
        value="${value//\"/}"
        [ -n "$value" ] || continue
        BENCHMARK_MENU_ARGS+=("--benchmark-spawn-total=$value")
        ;;
      8)
        read -r -p "Spawn VPS per spawn point: " value
        value="${value// /}"
        value="${value//\"/}"
        [ -n "$value" ] || continue
        BENCHMARK_MENU_ARGS+=("--benchmark-spawn-per-point-vps=$value")
        ;;
      *)
        echo "[Warning] Invalid load selection: $load"
        continue
        ;;
    esac
    break
  done

  echo
  echo "Optional overrides. Press Enter to keep config.txt value."
  read -r -p "Timed steps [current ${BENCHMARK_STEPS:-10000}]: " steps
  steps="${steps// /}"
  steps="${steps//\"/}"
  if [ -n "$steps" ]; then BENCHMARK_MENU_ARGS+=("--benchmark-steps=$steps"); fi

  read -r -p "Warmup steps [current ${BENCHMARK_WARMUP_STEPS:-1000}]: " warmup
  warmup="${warmup// /}"
  warmup="${warmup//\"/}"
  if [ -n "$warmup" ]; then BENCHMARK_MENU_ARGS+=("--benchmark-warmup-steps=$warmup"); fi

  read -r -p "Max agents [current ${BENCHMARK_MAX_AGENTS:-240000}]: " max_agents
  max_agents="${max_agents// /}"
  max_agents="${max_agents//\"/}"
  if [ -n "$max_agents" ]; then BENCHMARK_MENU_ARGS+=("--benchmark-max-agents=$max_agents"); fi

  case ",$BENCHMARK_MENU_ORDER," in
    *",sumo,"*)
      read -r -p "SUMO runner [auto/libsumo/cli, current ${SUMO_BENCHMARK_RUNNER:-auto}]: " sumo_runner
      sumo_runner="${sumo_runner// /}"
      sumo_runner="${sumo_runner//\"/}"
      if [ -n "$sumo_runner" ]; then BENCHMARK_MENU_ARGS+=("--sumo-runner=$sumo_runner"); fi
      ;;
  esac

  echo
  echo "[Info] Benchmark order: $BENCHMARK_MENU_ORDER"
  if [ "${#BENCHMARK_MENU_ARGS[@]}" -gt 0 ]; then
    echo "[Info] Extra args: ${BENCHMARK_MENU_ARGS[*]}"
  else
    echo "[Info] Extra args: (none)"
  fi
  echo
  return 0
}

run_benchmark_selector() {
  local rc
  if benchmark_selector; then
    rc=0
  else
    rc=$?
  fi
  if [ "$rc" -eq 2 ]; then
    echo "[Info] Benchmark selection cancelled."
    return 0
  fi
  if [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
  ensure_benchmark_backends_built "benchmark" "--benchmark-order=$BENCHMARK_MENU_ORDER" "${BENCHMARK_MENU_ARGS[@]}"
  launch_mode benchmark "--benchmark-order=$BENCHMARK_MENU_ORDER" "${BENCHMARK_MENU_ARGS[@]}"
}

show_menu() {
  echo
  echo "======================================================="
  echo "AVABM Launcher"
  echo "======================================================="
  echo "1. Turbo   - headless selected-backend batch run"
  echo "2. Visual  - OpenGL/Pygame selected-backend window run"
  echo "3. Build avabm package (CPU + CUDA)"
  echo "4. Build selected backend from SIM_BACKEND"
  echo "5. Build CUDA only (skip if up to date)"
  echo "6. Build CPU only"
  echo "7. Clean rebuild CUDA"
  echo "8. Hard clean + rebuild CUDA"
  echo "9. CUDA build status"
  echo "10. Exit"
  echo "11. Benchmark selector - choose CPU/CUDA/SUMO architecture(s)"
  echo
  printf 'Select [1-11]: '
}

cmd="${1:-}"
if [ -z "$cmd" ]; then
  show_menu
  read -r choice
  choice="$(printf '%s' "$choice" | tr -d '[:space:]\"')"
  case "$choice" in
    1|t|T|turbo|Turbo) set -- turbo ;;
    2|v|V|visual|Visual) set -- visual ;;
    3|a|A|all|All|build|Build|build-all|all-build) set -- build ;;
    4|b|B|selected|build-selected|selected-build) set -- build-selected ;;
    5|bc|build-cuda|cuda-build|cuda|CUDA) set -- build-cuda ;;
    6|cpu|CPU|build-cpu|cpu-build) set -- build-cpu ;;
    7|r|R|rebuild|Rebuild|clean|Clean) set -- rebuild ;;
    8|h|H|hardclean|Hardclean|cleanall|clean-all) set -- hardclean ;;
    9|s|S|status|Status|check|Check) set -- status ;;
    10|0|q|Q|exit|Exit) exit 0 ;;
    11|bench|Bench|benchmark|Benchmark) set -- benchmark-menu ;;
    *) echo "[Error] Invalid selection: $choice" >&2; exit 1 ;;
  esac
  cmd="$1"
  shift || true
else
  shift || true
fi

case "$cmd" in
  1|turbo|--turbo|headless|--headless)
    ensure_backend_built "$@"
    launch_mode turbo "$@"
    ;;
  2|visual|--visual|gui|--gui|window|--window)
    ensure_backend_built "$@"
    launch_mode visual "$@"
    ;;
  3|build|build-all|all-build|all)
    build_all
    ;;
  4|build-selected|selected-build|selected)
    ensure_backend_built "$@"
    ;;
  5|build-cuda|cuda-build|cuda)
    build_cuda auto
    ;;
  6|build-cpu|cpu-build|cpu)
    build_cpu
    ;;
  7|rebuild|clean)
    build_cuda clean
    ;;
  8|hardclean|cleanall|clean-all)
    build_cuda hardclean
    ;;
  9|status|check)
    activate_conda
    "$PYTHON_COMMAND" "$BUILD_HELPER" status
    ;;
  11|benchmark-menu|bench-menu|benchmark-select|bench-select|benchmark-architectures|bench-architectures)
    run_benchmark_selector
    ;;
  bench|benchmark)
    case "${1:-}" in
      menu|select|architectures|architecture-selector|arch-selector)
        shift || true
        run_benchmark_selector
        ;;
      *)
        ensure_benchmark_backends_built "$cmd" "$@"
        launch_mode benchmark "$cmd" "$@"
        ;;
    esac
    ;;
  benchmark-cpu|bench-cpu|benchmark-gpu|benchmark-cuda|bench-gpu|bench-cuda|benchmark-sumo|bench-sumo)
    ensure_benchmark_backends_built "$cmd" "$@"
    launch_mode benchmark "$cmd" "$@"
    ;;
  10|0|q|Q|exit|Exit)
    exit 0
    ;;
  *)
    echo "[Error] Unknown command: $cmd" >&2
    echo "Usage: ./run.sh [turbo|visual|benchmark|build|build-selected|build-cuda|build-cpu|rebuild|hardclean|status] [extra main.py args...]" >&2
    echo "       Benchmark shortcuts: ./run.sh benchmark cpu | gpu | sumo | both | all" >&2
    echo "       Benchmark load: ./run.sh benchmark gpu --with-sumo --benchmark-spawn-vps=100 --benchmark-max-agents=300000" >&2
    echo "       Numeric shortcuts are also accepted: 1..11"
    exit 1
    ;;
esac
