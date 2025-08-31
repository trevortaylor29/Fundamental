@echo off
setlocal EnableExtensions
cd /d "%~dp0"

rem ---- log everything ----
set "LOG=%cd%\run.log"
echo ============================================ > "%LOG%"
echo Fundamental: setup and launch               >> "%LOG%"
echo %DATE% %TIME%                               >> "%LOG%"
echo ============================================ >> "%LOG%"
echo (Writing detailed log to %LOG%)

rem ---- find python ----
where py >nul 2>nul
if %errorlevel%==0 (
  set "PY=py -3"
) else (
  where python >nul 2>nul
  if %errorlevel%==0 ( set "PY=python" ) else (
    echo [ERROR] Python 3 not found. >> "%LOG%"
    echo [ERROR] Python 3 not found.
    echo Install from https://www.python.org/downloads/ (check "Add Python to PATH") then run again.
    pause
    exit /b 1
  )
)

rem ---- create venv if missing ----
if not exist ".venv" (
  echo Creating virtual environment (.venv)...
  echo Creating virtual environment (.venv)... >> "%LOG%"
  %PY% -m venv .venv >> "%LOG%" 2>&1
  if %errorlevel% neq 0 (
    echo [ERROR] Failed to create venv. See run.log for details.
    pause
    exit /b 1
  )
)

rem ---- activate venv ----
call ".venv\Scripts\activate.bat"
if %errorlevel% neq 0 (
  echo [ERROR] Failed to activate venv. >> "%LOG%"
  echo [ERROR] Failed to activate venv.
  pause
  exit /b 1
)

rem ---- install deps ----
echo Upgrading pip... >> "%LOG%"
python -m pip install --upgrade pip >> "%LOG%" 2>&1

if exist requirements-lock.txt (
  echo Installing dependencies (locked)...
  echo Installing dependencies (locked)... >> "%LOG%"
  pip install -r requirements-lock.txt >> "%LOG%" 2>&1
) else (
  echo Installing dependencies...
  echo Installing dependencies... >> "%LOG%"
  pip install -r requirements.txt >> "%LOG%" 2>&1
)
if %errorlevel% neq 0 (
  echo [ERROR] Dependency install failed. See run.log for details.
  pause
  exit /b 1
)

rem ---- launch app ----
echo Starting app at http://localhost:8501
echo Starting app at http://localhost:8501 >> "%LOG%"
streamlit run streamlit_app.py >> "%LOG%" 2>&1

echo.
echo (You can close this window after you exit the app.)
pause
