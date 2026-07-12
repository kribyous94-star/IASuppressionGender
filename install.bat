@echo off
rem IASuppressionGender -- installation (Windows)
rem Usage :
rem   install.bat          ^<-- detection GPU automatique (recommande)
rem   install.bat --gpu    ^<-- forcer GPU NVIDIA (CUDA Toolkit requis)
rem   install.bat --cpu    ^<-- forcer CPU
chcp 65001 > nul
powershell.exe -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if %ERRORLEVEL% neq 0 pause
