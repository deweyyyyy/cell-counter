@echo off
rem Start the Cell Counter server on Windows. The first run creates a private Python environment.
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo First run: installing numpy and pillow...
  py -3 -m venv .venv || python -m venv .venv
  .venv\Scripts\python -m pip install -q --upgrade pip
  .venv\Scripts\python -m pip install -q -r requirements.txt
)
.venv\Scripts\python server.py %*
