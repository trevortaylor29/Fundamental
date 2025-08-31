@echo off
setlocal

echo === Fundamental: one-time setup ^& launch ===

REM --- Find Python (prefer 'py' launcher on Windows) ---
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    set "PY=python"
  ) else (
    echo.
    echo [ERROR] Python 3 not found.
    echo Please install Python 3.10+ from https://www.python.org/downloads/ (check "Add Python to PATH"),
    echo then double-click this file again.
    pause
    exit /b 1
  )
)

REM --- Create venv if missing ---
if not exist ".venv" (
  echo Creating virtual environment (.venv)...
  %PY% -m venv .venv
  if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

REM --- Activate venv ---
call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
  echo [ERROR] Failed to activate venv.
  pause
  exit /b 1
)

REM --- Upgrade pip and install deps ---
python -m pip install --upgrade pip
if exist requirements-lock.txt (
  echo Installing dependencies (locked)...
  pip install -r requirements-lock.txt
) else (
  echo Installing dependencies...
  pip install -r requirements.txt
)

if %errorlevel% neq 0 (
  echo [ERROR] Dependency install failed.
  pause
  exit /b 1
)

REM --- Launch app ---
echo Starting app at http://localhost:8501
streamlit run streamlit_app.py

echo.
echo (You can close this window after you exit the app.)
pause
