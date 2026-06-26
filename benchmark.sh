#!/usr/bin/env bash
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -eq 0 ]; then
  exec ./run.sh benchmark-menu
fi

case "${1,,}" in
  menu|select|architectures|architecture-selector|arch-selector)
    shift || true
    exec ./run.sh benchmark-menu "$@"
    ;;
  *)
    exec ./run.sh benchmark "$@"
    ;;
esac
