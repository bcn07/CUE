@echo off
REM One command per camera laptop (Windows):
REM   camera\run_camera.bat --server ws://DIRECTOR-IP:8000 --cam C --code JOIN-CODE
setlocal
set HERE=%~dp0
set VPY=%HERE%.venv\Scripts\python.exe
if not exist "%VPY%" (
  echo [cue-cam] creating venv ...
  py -3 -m venv "%HERE%.venv" || python -m venv "%HERE%.venv" || (echo [cue-cam] could not create a venv. Install Python 3 from python.org and tick "Add to PATH". & exit /b 1)
)
"%VPY%" -c "import cv2, websockets" >nul 2>&1
if errorlevel 1 (
  echo [cue-cam] installing opencv-python + websockets ...
  "%VPY%" -m pip install --quiet --upgrade pip
  "%VPY%" -m pip install --quiet -r "%HERE%requirements-camera.txt" || (echo [cue-cam] pip install failed, no internet? Retry when online. & exit /b 1)
)
"%VPY%" "%HERE%stream_camera.py" %*
