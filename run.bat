@echo off
rem IASuppressionGender — interface graphique (Windows)
rem Double-cliquez ou lancez depuis un terminal :
rem   run.bat                  (port 7860 par défaut)
rem   run.bat --port 8080      (ou -Port 8080)
rem   run.bat --no-browser     (ou -NoBrowser)
chcp 65001 > nul
powershell.exe -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if %ERRORLEVEL% neq 0 pause
