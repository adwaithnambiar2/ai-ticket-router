@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First create .venv and install requirements. See README.md.
  pause
  exit /b 1
)
.venv\Scripts\python.exe train.py
if errorlevel 1 exit /b 1
.venv\Scripts\python.exe app.py
