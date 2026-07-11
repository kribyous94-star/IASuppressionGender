#!/usr/bin/env bash
# IASuppressionGender — installation auto-contenue.
# Tout (venvs, modèles, caches) est installé DANS le dossier du projet :
# supprimer le dossier supprime toute l'installation.
#
# Usage :
#   ./install.sh          # CPU (défaut)
#   ./install.sh --gpu    # torch CUDA + onnxruntime-gpu
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1

info()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[erreur]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- pré-requis
command -v python3 >/dev/null || error "python3 introuvable."
python3 -c 'import venv' 2>/dev/null || error "module venv manquant (sudo apt install python3-venv)."
python3 - <<'EOF' || error "Python >= 3.10 requis."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

# Tous les caches restent dans le dossier
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$ROOT/venvs" "$ROOT/models" "$ROOT/.cache" "$YOLO_CONFIG_DIR"

make_venv() { # $1 = nom
    local dir="$ROOT/venvs/$1"
    if [[ ! -x "$dir/bin/python" ]]; then
        info "création du venv '$1'"
        python3 -m venv "$dir"
    fi
    "$dir/bin/pip" install --quiet --upgrade pip wheel setuptools
}

pipin() { # $1 = venv, reste = args pip install
    local venv="$1"; shift
    "$ROOT/venvs/$venv/bin/pip" install "$@"
}

# ---------------------------------------------------------------- venv core
make_venv core
info "venv core : dépendances (opencv, ffmpeg statique, …)"
pipin core -r "$ROOT/requirements/core.txt"
# force le téléchargement du binaire ffmpeg statique dans le venv
"$ROOT/venvs/core/bin/python" -c "import imageio_ffmpeg; print('  ffmpeg embarqué :', imageio_ffmpeg.get_ffmpeg_exe())"

# ---------------------------------------------------------------- venv face
make_venv face
info "venv face : insightface + onnxruntime"
if [[ $GPU -eq 1 ]]; then
    pipin face onnxruntime-gpu
else
    pipin face onnxruntime
fi
pipin face -r "$ROOT/requirements/face.txt" \
    || error "échec insightface (compilation). Installez les outils de build : sudo apt install build-essential python3-dev"

# ---------------------------------------------------------------- venv body
make_venv body
info "venv body : torch + ultralytics + open_clip"
if [[ $GPU -eq 1 ]]; then
    pipin body torch torchvision
else
    pipin body torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pipin body -r "$ROOT/requirements/body.txt"

# ---------------------------------------------------------------- modèles IA
info "modèles : InsightFace buffalo_l → models/insightface/"
"$ROOT/venvs/face/bin/python" - "$ROOT" <<'EOF'
import sys
from insightface.app import FaceAnalysis
root = sys.argv[1] + "/models/insightface"
app = FaceAnalysis(name="buffalo_l", root=root,
                   allowed_modules=["detection", "genderage"],
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("  buffalo_l OK")
EOF

info "modèles : YOLOv8s → models/yolo/"
mkdir -p "$ROOT/models/yolo"
( cd "$ROOT/models/yolo" && "$ROOT/venvs/body/bin/python" - <<'EOF'
from ultralytics import YOLO
YOLO("yolov8s.pt")  # télécharge dans le dossier courant si absent
print("  yolov8s OK")
EOF
)

info "modèles : CLIP ViT-B/32 (laion2b) → models/openclip/"
"$ROOT/venvs/body/bin/python" - "$ROOT" <<'EOF'
import sys
import open_clip
open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k",
    cache_dir=sys.argv[1] + "/models/openclip")
print("  CLIP OK")
EOF

info "installation terminée. Exemple : ./run.sh -i video.mp4 -g femme"
