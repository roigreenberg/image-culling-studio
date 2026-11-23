#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========================================================================"
echo "  ImageCulling Studio - macOS Installer (No Git or Python required)"
echo "========================================================================"
echo ""

# 1) Locate or bootstrap Astral uv (standalone binary, installs isolated Python 3.10)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "[1/5] Installing uv package manager (auto-manages Python 3.10)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: Could not install uv automatically. Please install Python 3.10 from https://www.python.org/downloads/ and retry."
  exit 1
fi

echo "[2/5] Ensuring Python 3.10 and creating virtual environment (.venv)..."
uv python install 3.10
uv venv .venv --python 3.10 --seed

echo "[3/5] Installing PyTorch (Apple Silicon MPS / CPU) + ONNX Runtime..."
uv pip install --python .venv/bin/python torch torchvision onnxruntime

echo "[4/5] Installing ImageCulling Studio dependencies..."
uv pip install --python .venv/bin/python -r requirements-base.txt

echo "[5/5] Downloading required Project Kestrel AI model weights (~480 MB)..."
.venv/bin/python setup_models.py || {
  echo "WARNING: Some model downloads failed. Retry later with: .venv/bin/python setup_models.py"
}

chmod +x Start_MacOS.command 2>/dev/null || true

echo ""
echo "========================================================================"
echo "  Installation complete!"
echo "  Double-click Start_MacOS.command to launch ImageCulling Studio."
echo "========================================================================"
read -r -p "Press Enter to close..."
