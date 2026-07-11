@echo off
rem IASuppressionGender -- installation (Windows)
rem Usage :
rem   install.bat          ^<-- CPU (defaut)
rem   install.bat --gpu    ^<-- GPU NVIDIA (CUDA Toolkit requis)
chcp 65001 > nul
powershell.exe -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
if %ERRORLEVEL% neq 0 pause
