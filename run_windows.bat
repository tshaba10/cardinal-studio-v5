@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Follow README.md to create the project environment first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run app.py %*
