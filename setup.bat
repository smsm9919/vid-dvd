@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
 echo Python launcher not found. Install Python 3.11+.
 pause
 exit /b 1
)
py -3.11 -m venv .venv
call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
pip install -r requirements.txt
echo Setup complete.
pause
