#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
CONFIG_FILE="config.txt"
load_config() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r raw || [ -n "$raw" ]; do
    case "$raw" in
      ''|'#'*|';'*) continue ;;
    esac
    case "$raw" in *=*) ;; *) continue ;; esac
    key="${raw%%=*}"
    val="${raw#*=}"
    key="$(printf '%s' "$key" | sed 's/[[:space:]]//g')"
    case "$key" in ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;; esac
    val="$(printf '%s' "$val" | sed 's/[[:space:]]*#.*$//; s/[[:space:]]*;.*$//; s/^[[:space:]]*//; s/[[:space:]]*$//; s/^"//; s/"$//')"
    export "$key=$val"
  done < "$file"
}
load_config "$CONFIG_FILE"
: "${PYTHON_COMMAND:=python}"
"$PYTHON_COMMAND" avabm_cuda/build_helper.py verify-run
rc=$?
if [ "$rc" -eq 2 ]; then
  echo "[Error] Compiled CUDA module is missing or stale. Run ./avabm_cuda/build.sh first."
  exit 1
elif [ "$rc" -ne 0 ]; then
  echo "[Error] CUDA build fingerprint check failed."
  exit 1
fi
exec "$PYTHON_COMMAND" main.py
