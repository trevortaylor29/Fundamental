@echo on
setlocal
cd /d "%~dp0"

REM Find Python (prefer the py launcher)
where py >nul 2>nul || echo (no py launcher)
where python >nul 2>nul || echo (no python on PATH)

REM Create venv if missing
if not exist ".venv\Scripts\activate.bat" (
  py -3 -m venv .venv 2>nul || python -m venv .venv
)

REM Activate venv
call ".venv\Scripts\activate.bat" || goto :fail

REM Install deps
python -m pip install --upgrade pip || goto :fail
if exist requirements-lock.txt (
  pip install -r requirements-lock.txt || goto :fail
) else (
  pip install -r requirements.txt || goto :fail
)

REM Launch
streamlit run streamlit_app.py
goto :eof

:fail
echo.
echo [ERROR] Something failed. Review the lines above.
pause
