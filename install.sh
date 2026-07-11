#!/usr/bin/env bash
# IASuppressionGender — installation auto-contenue.
# Tout (venvs, modèles, caches, libs CUDA) est installé DANS le dossier du
# projet : supprimer le dossier supprime toute l'installation.
#
# Usage :
#   ./install.sh          # CPU (défaut)
#   ./install.sh --gpu    # GPU NVIDIA : torch CUDA + onnxruntime-gpu
#                         # (libs CUDA installées via pip, dans les venvs)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU=0
[[ "${1:-}" == "--gpu" ]] && GPU=1

info()  { printf '\033[1;34m[install]\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m[attention]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[erreur]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- pré-requis
command -v python3 >/dev/null || error "python3 introuvable."
python3 -c 'import venv' 2>/dev/null || error "module venv manquant (sudo apt install python3-venv)."
python3 - <<'EOF' || error "Python >= 3.10 requis."
import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
if [[ $GPU -eq 1 ]]; then
    command -v nvidia-smi >/dev/null \
        || error "--gpu demandé mais aucun pilote NVIDIA détecté (nvidia-smi introuvable)."
    info "GPU détecté : $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
fi

# Tous les caches restent dans le dossier
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export HF_HOME="$ROOT/.cache/huggingface"
export TORCH_HOME="$ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOT/.cache/xdg"
export YOLO_CONFIG_DIR="$ROOT/.cache/ultralytics"
mkdir -p "$ROOT/venvs" "$ROOT/models" "$ROOT/.cache" "$ROOT/output" "$YOLO_CONFIG_DIR"

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

pipout() { # $1 = venv, reste = paquets à désinstaller (silencieux si absents)
    local venv="$1"; shift
    "$ROOT/venvs/$venv/bin/pip" uninstall -y -q "$@" >/dev/null 2>&1 || true
}

nvlibs() { # $1 = venv → chemins des libs CUDA installées via pip, joints par ':'
    find "$ROOT/venvs/$1" -type d -path '*/site-packages/nvidia/*/lib' 2>/dev/null | paste -sd: -
}

# ---------------------------------------------------------------- venv core
make_venv core
info "venv core : orchestration + interface (opencv, gradio, ffmpeg statique, …)"
pipin core -r "$ROOT/requirements/core.txt"
# force le téléchargement du binaire ffmpeg statique dans le venv
"$ROOT/venvs/core/bin/python" -c "import imageio_ffmpeg; print('  ffmpeg embarqué :', imageio_ffmpeg.get_ffmpeg_exe())"

# ---------------------------------------------------------------- venv face
make_venv face
info "venv face : insightface + onnxruntime"
# Les variantes CPU/GPU fournissent le même module Python : elles ne doivent
# jamais cohabiter, et une désinstallation partielle laisse un module cassé →
# on retire TOUJOURS les deux avant d'installer la bonne.
ORT_WANTED="onnxruntime"
[[ $GPU -eq 1 ]] && ORT_WANTED="onnxruntime-gpu"
ORT_STATE="$("$ROOT/venvs/face/bin/pip" list 2>/dev/null | grep -oE '^onnxruntime(-gpu)?' | sort | paste -sd+ -)"
if [[ "$ORT_STATE" != "$ORT_WANTED" ]]; then
    pipout face onnxruntime onnxruntime-gpu
    pipin face "$ORT_WANTED"
fi
if [[ $GPU -eq 1 ]]; then
    # libs CUDA 13 via pip — noms non suffixés (nouvelle convention NVIDIA),
    # sauf cudnn qui garde son suffixe -cu13
    pipin face \
        nvidia-cuda-runtime nvidia-cuda-nvrtc nvidia-cublas \
        nvidia-cudnn-cu13 nvidia-cufft nvidia-curand
fi
pipin face -r "$ROOT/requirements/face.txt" \
    || error "échec insightface (compilation). Installez les outils de build : sudo apt install build-essential python3-dev"

# ---------------------------------------------------------------- venv body
make_venv body
info "venv body : torch + ultralytics + open_clip"
TORCH_VER="$("$ROOT/venvs/body/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo aucune)"
if [[ $GPU -eq 1 ]]; then
    # une version +cpu déjà présente empêcherait pip d'installer la variante CUDA
    if [[ "$TORCH_VER" == *"+cpu"* ]]; then
        info "torch $TORCH_VER (CPU) présent → remplacement par la variante CUDA"
        pipout body torch torchvision
    fi
    pipin body torch torchvision
else
    if [[ "$TORCH_VER" != aucune && "$TORCH_VER" != *"+cpu"* ]]; then
        info "torch $TORCH_VER (CUDA) présent → remplacement par la variante CPU"
        pipout body torch torchvision
    fi
    pipin body torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
pipin body -r "$ROOT/requirements/body.txt"

if [[ $GPU -eq 1 ]]; then
    info "vérification CUDA (torch)…"
    "$ROOT/venvs/body/bin/python" - <<'EOF' || warn "torch ne voit pas le GPU : le détecteur 'body' tournera sur CPU"
import torch
assert torch.cuda.is_available()
print("  torch CUDA OK :", torch.cuda.get_device_name(0))
EOF
fi

# ---------------------------------------------------------------- modèles IA
info "modèles : InsightFace buffalo_l → models/insightface/"
LD_LIBRARY_PATH="$(nvlibs face)" "$ROOT/venvs/face/bin/python" - "$ROOT" <<'EOF'
import sys
from insightface.app import FaceAnalysis
root = sys.argv[1] + "/models/insightface"
app = FaceAnalysis(name="buffalo_l", root=root,
                   allowed_modules=["detection", "genderage"],
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("  buffalo_l OK")
EOF

if [[ $GPU -eq 1 ]]; then
    info "vérification CUDA (onnxruntime) : session d'inférence réelle…"
    if LD_LIBRARY_PATH="$(nvlibs face)" "$ROOT/venvs/face/bin/python" - "$ROOT" <<'EOF'
import glob, sys
import onnxruntime as ort
model = glob.glob(sys.argv[1] + "/models/insightface/models/buffalo_l/det_*.onnx")[0]
sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider"])
assert "CUDAExecutionProvider" in sess.get_providers(), sess.get_providers()
print("  onnxruntime-gpu OK :", sess.get_providers())
EOF
    then :; else
        warn "onnxruntime-gpu inutilisable sur cette machine → repli sur la version CPU"
        pipout face onnxruntime onnxruntime-gpu
        pipin face onnxruntime
    fi
fi

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

info "installation terminée."
info "  interface : ./run.sh"
info "  ligne de commande : ./cli.sh -i video.mp4 -g femme"
