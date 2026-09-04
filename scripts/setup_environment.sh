#!/usr/bin/env bash
# Create a standalone Isaac Lab environment for MIRRORME RL Train.
# Usage: ./scripts/setup_environment.sh [environment-name] [--skip-contract]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="bpx_train"
RUN_CONTRACT=true
ENV_NAME_SET=false

progress() {
  echo
  echo "================================================================"
  echo "[$1/6] $2"
  echo "================================================================"
}

for argument in "$@"; do
  case "$argument" in
    --skip-contract) RUN_CONTRACT=false ;;
    -h|--help)
      echo "Usage: ./scripts/setup_environment.sh [environment-name] [--skip-contract]"
      exit 0
      ;;
    -*)
      echo "Unknown option: $argument" >&2
      exit 2
      ;;
    *)
      if [[ "$ENV_NAME_SET" == "true" ]]; then
        echo "Only one environment name may be supplied." >&2
        exit 2
      fi
      ENV_NAME="$argument"
      ENV_NAME_SET=true
      ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Conda is required. Install Miniconda or Anaconda, then run this script again." >&2
  exit 1
fi

if [[ ! -f "$ROOT/resources/BPX/bpx/urdf/bpx.urdf" ]]; then
  echo "The BPX asset submodule is missing." >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 1
fi

if conda run --no-capture-output -n "$ENV_NAME" python --version >/dev/null 2>&1; then
  echo "Conda environment '$ENV_NAME' already exists." >&2
  echo "For safety, setup only creates a new environment and never reuses or modifies an existing one." >&2
  echo "Choose a new name, for example: ./bpx.sh setup bpx_train_new" >&2
  exit 1
fi

progress 1 "Creating new Conda environment: $ENV_NAME"
conda create -n "$ENV_NAME" python=3.11 -y

progress 2 "Preparing Python build tools"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --upgrade pip
# Isaac Lab 2.3.0 depends on flatdict 4.0.1. Its legacy build script imports
# pkg_resources, which is no longer bundled with recent setuptools releases.
# Install a compatible build toolchain and build flatdict before resolving the
# rest of the Isaac Lab dependencies.
conda run --no-capture-output -n "$ENV_NAME" python -m pip install "setuptools<81" wheel
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --no-build-isolation flatdict==4.0.1

progress 3 "Downloading and installing Isaac Lab, Isaac Sim, and RSL-RL (this can take several minutes)"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install --verbose \
  "isaaclab[isaacsim,rsl-rl]==2.3.0" \
  --progress-bar on \
  --extra-index-url https://pypi.nvidia.com

progress 4 "Installing CUDA-enabled PyTorch"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -U \
  torch==2.7.0 torchvision==0.22.0 \
  --progress-bar on \
  --index-url https://download.pytorch.org/whl/cu128

progress 5 "Installing MIRRORME RL Train"
conda run --no-capture-output -n "$ENV_NAME" python -m pip install -e "$ROOT/source/mirrorme_rl_train"

progress 6 "Checking Python, GPU, and BPX runtime"
conda run --no-capture-output -n "$ENV_NAME" python -c \
  "import isaaclab, isaacsim, torch; assert torch.cuda.is_available(), 'CUDA GPU is unavailable'; print('Isaac Lab, Isaac Sim, and CUDA are ready.')"

if [[ "$RUN_CONTRACT" == "true" ]]; then
  echo "Starting the BPX runtime contract check. Isaac Sim may take a short time to launch."
  cd "$ROOT"
  conda run --no-capture-output -n "$ENV_NAME" ./bpx.sh contract --headless
fi

echo
echo "Setup complete. Activate it with: conda activate $ENV_NAME"
