#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EGOEXOR_DIR="${SCRIPT_DIR}/EgoExOR"

echo "=== EgoExOR Evaluation Setup ==="

if [ ! -d "${EGOEXOR_DIR}/.git" ]; then
    echo "[1/3] Cloning EgoExOR repository..."
    git clone https://github.com/ardamamur/EgoExOR.git "${EGOEXOR_DIR}"
else
    echo "[1/3] EgoExOR repository already exists, skipping clone."
fi

echo "[2/3] Installing LLaVA (editable)..."
uv pip install -e "${EGOEXOR_DIR}/scene_graph_generation/LLaVA"

echo "[3/3] Installing flash-attn (requires CUDA + nvcc)..."
if command -v nvcc &>/dev/null; then
    uv pip install flash-attn --no-build-isolation 2>/dev/null || \
        echo "WARNING: flash-attn install failed. Install manually on CUDA machine."
else
    echo "WARNING: nvcc not found. Skipping flash-attn (required on CUDA machine)."
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Run: python download.py --miss-only"
echo "  2. Run: python evaluate.py --model_path data/model --test_json data/test_samples.json --hdf5_path data/egoexor_miss.h5"
