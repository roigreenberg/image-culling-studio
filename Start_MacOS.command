#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "ImageCulling Studio is not installed yet."
  echo "Please double-click Install_MacOS.command first!"
  read -r -p "Press Enter to close..."
  exit 1
fi

echo "Starting ImageCulling Studio at http://127.0.0.1:8765 ..."
(sleep 2 && open "http://127.0.0.1:8765") &
.venv/bin/python run_webapp.py --host 127.0.0.1 --port 8765
