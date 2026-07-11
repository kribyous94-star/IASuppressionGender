# IASuppressionGender — lance l'interface graphique (Windows, 100 % hors ligne).
# Options : -Port N, -NoBrowser
# Pour la ligne de commande : .\cli.ps1 -i video.mp4 -g femme

param(
    [int]   $Port      = 7860,
    [switch]$NoBrowser
)

$ROOT = $PSScriptRoot
$PY   = "$ROOT\venvs\core\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "[erreur] venvs manquants : lancez d'abord .\install.ps1" -ForegroundColor Red
    exit 1
}

# Caches confinés au dossier + mode strictement hors ligne
$env:HF_HOME                  = "$ROOT\.cache\huggingface"
$env:TORCH_HOME               = "$ROOT\.cache\torch"
$env:YOLO_CONFIG_DIR          = "$ROOT\.cache\ultralytics"
$env:GRADIO_TEMP_DIR          = "$ROOT\.cache\gradio"

foreach ($d in @($env:YOLO_CONFIG_DIR, $env:GRADIO_TEMP_DIR, "$ROOT\output")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force $d | Out-Null }
}

$env:HF_HUB_OFFLINE           = "1"
$env:TRANSFORMERS_OFFLINE     = "1"
$env:YOLO_OFFLINE             = "1"
$env:GRADIO_ANALYTICS_ENABLED = "0"

$pyArgs = @("$ROOT\src\ui.py", "--port", $Port)
if ($NoBrowser) { $pyArgs += "--no-browser" }

& $PY @pyArgs
