# IASuppressionGender — ligne de commande (Windows, 100 % hors ligne).
# Exemple : .\cli.ps1 -i video.mp4 -g femme
# Pour l'interface graphique : .\run.ps1

$ROOT = $PSScriptRoot
$PY   = "$ROOT\venvs\core\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "[erreur] venvs manquants : lancez d'abord .\install.ps1" -ForegroundColor Red
    exit 1
}

# Caches confinés au dossier + mode strictement hors ligne
$env:HF_HOME              = "$ROOT\.cache\huggingface"
$env:TORCH_HOME           = "$ROOT\.cache\torch"
$env:YOLO_CONFIG_DIR      = "$ROOT\.cache\ultralytics"

if (-not (Test-Path $env:YOLO_CONFIG_DIR)) {
    New-Item -ItemType Directory -Force $env:YOLO_CONFIG_DIR | Out-Null
}

$env:HF_HUB_OFFLINE       = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:YOLO_OFFLINE         = "1"

& $PY "$ROOT\src\main.py" @args
