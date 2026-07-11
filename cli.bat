@echo off
rem IASuppressionGender — ligne de commande (Windows)
rem Exemple : cli.bat -i video.mp4 -g femme
rem Pour l'interface graphique : run.bat
powershell.exe -ExecutionPolicy Bypass -File "%~dp0cli.ps1" %*
if %ERRORLEVEL% neq 0 pause
