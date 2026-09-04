#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

usage() {
  cat <<'EOF'
Usage:
  ./bpx.sh install
  ./bpx.sh contract [Isaac Lab args]
  ./bpx.sh train [train args]
  ./bpx.sh play --checkpoint PATH [play args]
  ./bpx.sh export --checkpoint PATH [export args]
EOF
}

command="${1:-}"
[[ $# -gt 0 ]] && shift || true
case "$command" in
  install)
    python -m pip install -e source/mirrorme_rl_train
    ;;
  contract)
    python scripts/inspect_contract.py "$@"
    ;;
  train)
    python scripts/train.py "$@"
    ;;
  play)
    python scripts/play.py "$@"
    ;;
  export)
    python scripts/export_policy.py "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac
