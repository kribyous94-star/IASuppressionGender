#!/usr/bin/env bash
# IASuppressionGender — lancement (100 % hors ligne).
# Exemple : ./run.sh -i video.mp4 -g femme
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$ROOT/venvs/core/bin/python"

[[ -x "$PY" ]] || { echo "venvs manquants : lancez d'abord ./install.sh" >&2; exit 1; }

# Caches confinés au dossier + mode strictement hors ligne
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$YOLO_CONFIG_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export YOLO_OFFLINE=1

exec "$PY" "$ROOT/src/main.py" "$@"
