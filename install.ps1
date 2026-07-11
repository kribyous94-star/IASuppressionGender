# IASuppressionGender — installation auto-contenue (Windows).
# Tout (venvs, modèles, caches) est installé DANS le dossier du projet :
# supprimer le dossier supprime toute l'installation.
#
# Usage :
#   .\install.ps1          # CPU (défaut)
#   .\install.ps1 --gpu    # GPU NVIDIA : torch CUDA + onnxruntime-gpu
#
# Pré-requis Windows :
#   - Python 3.10+ : https://python.org  (cocher "Add Python to PATH")
#   - Pour insightface (compilation C++) :
#       Visual Studio Build Tools avec "Développement Desktop en C++"
#       https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/
#   - Pour GPU uniquement : CUDA Toolkit 12.x + cuDNN installés sur le système
#       https://developer.nvidia.com/cuda-downloads
#       https://developer.nvidia.com/cudnn

param([switch]$gpu)

$ROOT = $PSScriptRoot

function info  { Write-Host "[install] $args" -ForegroundColor Cyan }
function warn  { Write-Host "[attention] $args" -ForegroundColor Yellow }
function err   { Write-Host "[erreur] $args" -ForegroundColor Red; exit 1 }

function make_venv {
    $name = $args[0]
    $dir  = "$ROOT\venvs\$name"
    if (-not (Test-Path "$dir\Scripts\python.exe")) {
        info "création du venv '$name'"
        python -m venv $dir
        if ($LASTEXITCODE -ne 0) { err "Impossible de créer le venv '$name'." }
    }
    & "$dir\Scripts\pip.exe" install --quiet --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { err "Mise à jour de pip échouée (venv '$name')." }
}

function pipin {
    $venv    = $args[0]
    $pipArgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    & "$ROOT\venvs\$venv\Scripts\pip.exe" install @pipArgs
    if ($LASTEXITCODE -ne 0) { err "pip install a échoué (venv '$venv')." }
}

function pipout {
    $venv = $args[0]
    $pkgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    if ($pkgs.Count -gt 0) {
        & "$ROOT\venvs\$venv\Scripts\pip.exe" uninstall -y -q @pkgs 2>$null
    }
}

# ---------------------------------------------------------------- pré-requis
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pyCmd) {
    err "python introuvable. Installez Python 3.10+ depuis https://python.org (cochez 'Add Python to PATH')."
}

python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    $ver = python --version 2>&1
    err "Python >= 3.10 requis. Version actuelle : $ver"
}

python -c "import venv" 2>$null
if ($LASTEXITCODE -ne 0) { err "module venv manquant. Réinstallez Python depuis https://python.org." }

if ($gpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        err "--gpu demandé mais nvidia-smi introuvable. Installez le pilote NVIDIA et le CUDA Toolkit."
    }
    $gpuName = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    info "GPU détecté : $gpuName"
}

# Tous les caches restent dans le dossier du projet
$env:PIP_CACHE_DIR   = "$ROOT\.cache\pip"
$env:HF_HOME         = "$ROOT\.cache\huggingface"
$env:TORCH_HOME      = "$ROOT\.cache\torch"
$env:YOLO_CONFIG_DIR = "$ROOT\.cache\ultralytics"

foreach ($d in @("$ROOT\venvs", "$ROOT\models", "$ROOT\.cache", "$ROOT\output", $env:YOLO_CONFIG_DIR)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
}

# ---------------------------------------------------------------- venv core
make_venv core
info "venv core : orchestration + interface (opencv, gradio, ffmpeg statique, ...)"
pipin core -r "$ROOT\requirements\core.txt"
& "$ROOT\venvs\core\Scripts\python.exe" -c "import imageio_ffmpeg; print('  ffmpeg embarque :', imageio_ffmpeg.get_ffmpeg_exe())"

# ---------------------------------------------------------------- venv face
make_venv face
info "venv face : insightface + onnxruntime"

$ORT_WANTED = if ($gpu) { "onnxruntime-gpu" } else { "onnxruntime" }
$wrongOrt   = if ($gpu) { "onnxruntime" } else { "onnxruntime-gpu" }

& "$ROOT\venvs\face\Scripts\pip.exe" show $wrongOrt  2>$null | Out-Null
$wrongInstalled = ($LASTEXITCODE -eq 0)
& "$ROOT\venvs\face\Scripts\pip.exe" show $ORT_WANTED 2>$null | Out-Null
$rightInstalled = ($LASTEXITCODE -eq 0)

if ($wrongInstalled -or -not $rightInstalled) {
    pipout face onnxruntime onnxruntime-gpu
    pipin  face $ORT_WANTED
}

# Sur Windows, CUDA est fourni par le Toolkit système (pas par des paquets pip
# nvidia-* comme sur Linux). onnxruntime-gpu le détecte automatiquement.

info "venv face : installation d'insightface (compilation C++ requise si pas de wheel binaire)"
& "$ROOT\venvs\face\Scripts\pip.exe" install -r "$ROOT\requirements\face.txt"
if ($LASTEXITCODE -ne 0) {
    err (@"
Echec de la compilation d'insightface.
Installez les outils Build C++ puis relancez .\install.ps1 :
  https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/
  (sélectionnez "Développement Desktop en C++")
"@)
}

# ---------------------------------------------------------------- venv body
make_venv body
info "venv body : torch + ultralytics + open_clip"

$torchVer = (& "$ROOT\venvs\body\Scripts\python.exe" -c "import torch; print(torch.__version__)" 2>$null)
if (-not $torchVer) { $torchVer = "aucune" }

if ($gpu) {
    # Détection automatique de la version CUDA pour choisir l'index PyTorch correct
    $cudaIndex = "cu124"  # valeur par défaut ; modifiez si nécessaire (cu118, cu121, cu124)
    $smiOut = (nvidia-smi 2>$null) -join "`n"
    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $cudaMaj = [int]$Matches[1]
        $cudaMin = [int]$Matches[2]
        if     ($cudaMaj -lt 12)                     { $cudaIndex = "cu118" }
        elseif ($cudaMaj -eq 12 -and $cudaMin -lt 4) { $cudaIndex = "cu121" }
        else                                          { $cudaIndex = "cu124" }
        info "CUDA $cudaMaj.$cudaMin détecté → index PyTorch : $cudaIndex"
    } else {
        warn "Version CUDA non détectée dans la sortie nvidia-smi ; utilisation de $cudaIndex par défaut."
    }

    if ($torchVer -like "*+cpu*") {
        info "torch $torchVer (CPU) présent → remplacement par la variante CUDA"
        pipout body torch torchvision
    }
    pipin body torch torchvision --index-url "https://download.pytorch.org/whl/$cudaIndex"
} else {
    if ($torchVer -ne "aucune" -and $torchVer -notlike "*+cpu*") {
        info "torch $torchVer (CUDA) présent → remplacement par la variante CPU"
        pipout body torch torchvision
    }
    pipin body torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
}
pipin body -r "$ROOT\requirements\body.txt"

if ($gpu) {
    info "vérification CUDA (torch)..."
    $cudaCheck = @'
import torch, sys
if not torch.cuda.is_available():
    print("  [warn] torch.cuda.is_available() == False")
    sys.exit(1)
print("  torch CUDA OK :", torch.cuda.get_device_name(0))
'@
    & "$ROOT\venvs\body\Scripts\python.exe" -c $cudaCheck
    if ($LASTEXITCODE -ne 0) {
        warn "torch ne voit pas le GPU : le détecteur 'body' tournera sur CPU."
    }
}

# ---------------------------------------------------------------- modèles IA
info "modèles : InsightFace buffalo_l → models\insightface\"
$insightScript = @'
import sys
from insightface.app import FaceAnalysis
root = sys.argv[1] + "/models/insightface"
app = FaceAnalysis(name="buffalo_l", root=root,
                   allowed_modules=["detection", "genderage"],
                   providers=["CPUExecutionProvider"])
app.prepare(ctx_id=-1, det_size=(640, 640))
print("  buffalo_l OK")
'@
& "$ROOT\venvs\face\Scripts\python.exe" -c $insightScript $ROOT
if ($LASTEXITCODE -ne 0) { err "Echec du téléchargement du modèle InsightFace buffalo_l." }

if ($gpu) {
    info "vérification CUDA (onnxruntime) : session d'inférence réelle..."
    $ortCudaScript = @'
import glob, sys
import onnxruntime as ort
model = glob.glob(sys.argv[1] + "/models/insightface/models/buffalo_l/det_*.onnx")[0]
sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider"])
assert "CUDAExecutionProvider" in sess.get_providers(), sess.get_providers()
print("  onnxruntime-gpu OK :", sess.get_providers())
'@
    & "$ROOT\venvs\face\Scripts\python.exe" -c $ortCudaScript $ROOT
    if ($LASTEXITCODE -ne 0) {
        warn "onnxruntime-gpu inutilisable sur cette machine → repli sur la version CPU."
        pipout face onnxruntime onnxruntime-gpu
        pipin  face onnxruntime
    }
}

info "modèles : YOLOv8s → models\yolo\"
if (-not (Test-Path "$ROOT\models\yolo")) {
    New-Item -ItemType Directory -Force "$ROOT\models\yolo" | Out-Null
}
Push-Location "$ROOT\models\yolo"
& "$ROOT\venvs\body\Scripts\python.exe" -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); print('  yolov8s OK')"
$yoloCode = $LASTEXITCODE
Pop-Location
if ($yoloCode -ne 0) { err "Echec du téléchargement de YOLOv8s." }

info "modèles : CLIP ViT-B/32 (laion2b) → models\openclip\"
$clipScript = @'
import sys
import open_clip
open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="laion2b_s34b_b79k",
    cache_dir=sys.argv[1] + "/models/openclip")
print("  CLIP OK")
'@
& "$ROOT\venvs\body\Scripts\python.exe" -c $clipScript $ROOT
if ($LASTEXITCODE -ne 0) { err "Echec du téléchargement du modèle CLIP." }

info "installation terminée."
info "  interface         : .\run.ps1   (ou double-cliquez run.bat)"
info "  ligne de commande : .\cli.ps1 -i video.mp4 -g femme   (ou cli.bat)"
