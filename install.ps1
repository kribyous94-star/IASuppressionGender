# IASuppressionGender -- installation auto-contenue (Windows).
# Tout (venvs, modeles, caches) est installe DANS le dossier du projet :
# supprimer le dossier supprime toute l'installation.
#
# Usage :
#   .\install.ps1           # detection GPU automatique (recommande)
#   .\install.ps1 --cpu     # forcer CPU meme si un GPU NVIDIA est present
#   .\install.ps1 --gpu     # forcer GPU (echec si nvidia-smi absent)
#
# Pre-requis Windows :
#   - Python 3.10+ : https://python.org  (cocher "Add Python to PATH")
#   - Pour insightface (compilation C++) :
#       Visual Studio Build Tools avec "Developpement Desktop en C++"
#       https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/
#   - Pour GPU : CUDA Toolkit 12.x + pilote NVIDIA recents
#       https://developer.nvidia.com/cuda-downloads

param([switch]$gpu, [switch]$cpu)

$ROOT = $PSScriptRoot

# UTF-8 dans la console CMD (chcp 65001 cote bat + OutputEncoding cote PS)
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8

function info  { Write-Host "[install] $args" -ForegroundColor Cyan }
function warn  { Write-Host "[attention] $args" -ForegroundColor Yellow }
function err   { Write-Host "[erreur] $args" -ForegroundColor Red; exit 1 }

function make_venv {
    $name = $args[0]
    $dir  = "$ROOT\venvs\$name"
    if (-not (Test-Path "$dir\Scripts\python.exe")) {
        info "creation du venv '$name'"
        python -m venv $dir
        if ($LASTEXITCODE -ne 0) { err "Impossible de creer le venv '$name'." }
    }
    & "$dir\Scripts\python.exe" -m pip install --quiet --upgrade pip wheel "setuptools<82"
    if ($LASTEXITCODE -ne 0) { err "Mise a jour de pip echouee (venv '$name')." }
}

function pipin {
    $venv    = $args[0]
    $pipArgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    & "$ROOT\venvs\$venv\Scripts\pip.exe" install @pipArgs
    if ($LASTEXITCODE -ne 0) { err "pip install a echoue (venv '$venv')." }
}

function pipout {
    $venv = $args[0]
    $pkgs = if ($args.Count -gt 1) { $args[1..($args.Count - 1)] } else { @() }
    if ($pkgs.Count -gt 0) {
        & "$ROOT\venvs\$venv\Scripts\pip.exe" uninstall -y -q @pkgs 2>$null
    }
}

# ---------------------------------------------------------------- pre-requis
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
if ($LASTEXITCODE -ne 0) { err "module venv manquant. Reinstallez Python depuis https://python.org." }

# ---------------------------------------------------------------- detection GPU
# Priorite : --gpu force GPU, --cpu force CPU, sinon auto-detection
if ($cpu) {
    $gpu = $false
    info "mode CPU force par --cpu."
} elseif ($gpu) {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        err "--gpu demande mais nvidia-smi introuvable. Installez le pilote NVIDIA et CUDA Toolkit."
    }
} else {
    # Auto-detection : GPU si nvidia-smi repond avec succes
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($smi) {
        nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null
        $gpu = ($LASTEXITCODE -eq 0)
    } else {
        $gpu = $false
    }
    if ($gpu) { info "GPU NVIDIA detecte -> installation variante CUDA." }
    else       { info "Pas de GPU NVIDIA detecte -> installation variante CPU." }
}

if ($gpu) {
    $gpuName = (nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | Select-Object -First 1)
    info "GPU : $gpuName"
}

# Tous les caches restent dans le dossier du projet
$env:PIP_CACHE_DIR   = "$ROOT\.cache\pip"
$env:HF_HOME         = "$ROOT\.cache\huggingface"
$env:TORCH_HOME      = "$ROOT\.cache\torch"
$env:YOLO_CONFIG_DIR = "$ROOT\.cache\ultralytics"

# Preferer les wheels binaires precompiles pour eviter la compilation
# depuis les sources (risque de blocage par Windows Defender dans %TEMP%).
$env:PIP_PREFER_BINARY = "1"

foreach ($d in @("$ROOT\venvs", "$ROOT\models", "$ROOT\.cache", "$ROOT\output", $env:YOLO_CONFIG_DIR)) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
}

# ---------------------------------------------------------------- venv core
make_venv core
info "venv core : orchestration + interface (opencv, gradio, ffmpeg statique, ...)"

# numpy : purge le sdist en cache puis force le wheel binaire.
# Sans ca, Meson compile depuis les sources -> Defender bloque le binaire de test
# dans %TEMP% (faux positif Win64:MalwareX-gen sur sanitycheckcpp.exe).
& "$ROOT\venvs\core\Scripts\pip.exe" cache remove numpy 2>$null
& "$ROOT\venvs\core\Scripts\pip.exe" install --only-binary :all: "numpy>=1.24"
if ($LASTEXITCODE -ne 0) {
    err "Impossible d'installer le wheel numpy. Verifiez votre version de Python (3.10-3.14 requis)."
}

pipin core -r "$ROOT\requirements\core.txt"
& "$ROOT\venvs\core\Scripts\python.exe" -c "import imageio_ffmpeg; print('  ffmpeg embarque :', imageio_ffmpeg.get_ffmpeg_exe())"

# ---------------------------------------------------------------- venv face
make_venv face
info "venv face : insightface + onnxruntime"

# Protection numpy binaire (meme raison que pour venv core)
& "$ROOT\venvs\face\Scripts\pip.exe" cache remove numpy 2>$null
& "$ROOT\venvs\face\Scripts\pip.exe" install --only-binary :all: "numpy>=1.24"
if ($LASTEXITCODE -ne 0) { err "Impossible d'installer le wheel numpy (venv 'face')." }

# Installer la bonne variante onnxruntime (appel direct, pas via pipin,
# pour eviter un bug de splatting PS5.1 avec variable scalaire)
$ORT_WANTED = if ($gpu) { "onnxruntime-gpu" } else { "onnxruntime" }
$ORT_WRONG  = if ($gpu) { "onnxruntime"     } else { "onnxruntime-gpu" }
& "$ROOT\venvs\face\Scripts\pip.exe" uninstall -y -q $ORT_WRONG 2>$null
& "$ROOT\venvs\face\Scripts\pip.exe" install $ORT_WANTED
if ($LASTEXITCODE -ne 0) { err "Impossible d'installer $ORT_WANTED (venv 'face')." }

# Sur Windows, CUDA est fourni par le Toolkit systeme (pas par des paquets pip
# nvidia-* comme sur Linux). onnxruntime-gpu le detecte automatiquement.

info "venv face : installation d'insightface (compilation C++ requise si pas de wheel binaire)"
& "$ROOT\venvs\face\Scripts\pip.exe" install -r "$ROOT\requirements\face.txt"
if ($LASTEXITCODE -ne 0) {
    err (@"
Echec de la compilation d'insightface.
Installez les outils Build C++ puis relancez .\install.ps1 :
  https://visualstudio.microsoft.com/fr/visual-cpp-build-tools/
  (selectionnez "Developpement Desktop en C++")
"@)
}

# ---------------------------------------------------------------- venv body
make_venv body
info "venv body : torch + ultralytics + open_clip"

$torchVer = (& "$ROOT\venvs\body\Scripts\python.exe" -c "import torch; print(torch.__version__)" 2>$null)
if (-not $torchVer) { $torchVer = "aucune" }

if ($gpu) {
    # Detection automatique de la version CUDA pour choisir l'index PyTorch correct.
    # La version affichee par nvidia-smi est celle du pilote (max supportee) ;
    # on choisit le wheel le plus recent disponible inferieur ou egal.
    # Mapping : < 12 -> cu118 | 12.0-12.3 -> cu121 | 12.4-12.5 -> cu124
    #           12.6-12.7 -> cu126 | 12.8+ / 13.x -> cu128
    $cudaIndex = "cu128"  # valeur par defaut (CUDA 12.8+ / 13.x)
    $smiOut = (nvidia-smi 2>$null) -join "`n"
    if ($smiOut -match "CUDA Version:\s*(\d+)\.(\d+)") {
        $cudaMaj = [int]$Matches[1]
        $cudaMin = [int]$Matches[2]
        if     ($cudaMaj -lt 12)                     { $cudaIndex = "cu118" }
        elseif ($cudaMaj -eq 12 -and $cudaMin -lt 4) { $cudaIndex = "cu121" }
        elseif ($cudaMaj -eq 12 -and $cudaMin -lt 6) { $cudaIndex = "cu124" }
        elseif ($cudaMaj -eq 12 -and $cudaMin -lt 8) { $cudaIndex = "cu126" }
        else                                          { $cudaIndex = "cu128" }
        info "CUDA $cudaMaj.$cudaMin detecte -> index PyTorch : $cudaIndex"
    } else {
        warn "Version CUDA non detectee dans nvidia-smi ; utilisation de $cudaIndex par defaut."
    }

    if ($torchVer -like "*+cpu*") {
        info "torch $torchVer (CPU) present -> remplacement par la variante CUDA"
        pipout body torch torchvision
    }
    & "$ROOT\venvs\body\Scripts\pip.exe" install torch torchvision --index-url "https://download.pytorch.org/whl/$cudaIndex"
    if ($LASTEXITCODE -ne 0) {
        warn "Aucun wheel torch CUDA ($cudaIndex) pour cette version de Python."
        warn "Le detecteur 'body' tournera sur CPU ; onnxruntime-gpu utilisera quand meme le GPU pour 'face'."
        pipout body torch torchvision
        & "$ROOT\venvs\body\Scripts\pip.exe" install torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
        if ($LASTEXITCODE -ne 0) { err "Impossible d'installer torch (CPU ni CUDA)." }
    }
} else {
    if ($torchVer -ne "aucune" -and $torchVer -notlike "*+cpu*") {
        info "torch $torchVer (CUDA) present -> remplacement par la variante CPU"
        pipout body torch torchvision
    }
    pipin body torch torchvision --index-url "https://download.pytorch.org/whl/cpu"
}
pipin body -r "$ROOT\requirements\body.txt"

if ($gpu) {
    info "verification CUDA (torch)..."
    $cudaCheck = @'
import torch, sys
if not torch.cuda.is_available():
    print('  [warn] torch.cuda.is_available() == False')
    sys.exit(1)
print('  torch CUDA OK :', torch.cuda.get_device_name(0))
'@
    & "$ROOT\venvs\body\Scripts\python.exe" -c $cudaCheck
    if ($LASTEXITCODE -ne 0) {
        warn "torch ne voit pas le GPU : le detecteur 'body' tournera sur CPU."
    }
}

# ---------------------------------------------------------------- modeles IA
info "modeles : InsightFace buffalo_l -> models\insightface\"
$insightScript = @'
import sys
from insightface.app import FaceAnalysis
root = sys.argv[1] + '/models/insightface'
app = FaceAnalysis(name='buffalo_l', root=root,
                   allowed_modules=['detection', 'genderage'],
                   providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640, 640))
print('  buffalo_l OK')
'@
& "$ROOT\venvs\face\Scripts\python.exe" -c $insightScript $ROOT
if ($LASTEXITCODE -ne 0) { err "Echec du telechargement du modele InsightFace buffalo_l." }

if ($gpu) {
    info "verification CUDA (onnxruntime) : session d'inference reelle..."
    $ortCudaScript = @'
import glob, sys
import onnxruntime as ort
model = glob.glob(sys.argv[1] + '/models/insightface/models/buffalo_l/det_*.onnx')[0]
sess = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
assert 'CUDAExecutionProvider' in sess.get_providers(), sess.get_providers()
print('  onnxruntime-gpu OK :', sess.get_providers())
'@
    & "$ROOT\venvs\face\Scripts\python.exe" -c $ortCudaScript $ROOT 2>$null
    if ($LASTEXITCODE -ne 0) {
        warn "onnxruntime-gpu inutilisable -> repli sur onnxruntime CPU."
        warn "Pour activer le GPU sur InsightFace : CUDA Toolkit 13 + cuDNN 9 doivent etre installes."
        warn "Voir : https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html"
        & "$ROOT\venvs\face\Scripts\pip.exe" uninstall -y -q onnxruntime-gpu onnxruntime 2>$null
        & "$ROOT\venvs\face\Scripts\pip.exe" install onnxruntime
        if ($LASTEXITCODE -ne 0) { err "Impossible d'installer onnxruntime (repli CPU)." }
    }
}

info "modeles : YOLOv8s -> models\yolo\"
if (-not (Test-Path "$ROOT\models\yolo")) {
    New-Item -ItemType Directory -Force "$ROOT\models\yolo" | Out-Null
}
Push-Location "$ROOT\models\yolo"
& "$ROOT\venvs\body\Scripts\python.exe" -c "from ultralytics import YOLO; YOLO('yolov8s.pt'); print('  yolov8s OK')"
$yoloCode = $LASTEXITCODE
Pop-Location
if ($yoloCode -ne 0) { err "Echec du telechargement de YOLOv8s." }

info "modeles : CLIP ViT-B/32 (laion2b) -> models\openclip\"
$clipScript = @'
import sys
import open_clip
open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k',
    cache_dir=sys.argv[1] + '/models/openclip')
print('  CLIP OK')
'@
& "$ROOT\venvs\body\Scripts\python.exe" -c $clipScript $ROOT
if ($LASTEXITCODE -ne 0) { err "Echec du telechargement du modele CLIP." }

info "installation terminee."
info "  interface         : .\run.ps1   (ou double-cliquez run.bat)"
info "  ligne de commande : .\cli.ps1 -i video.mp4 -g femme   (ou cli.bat)"
