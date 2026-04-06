@echo off

REM Change to script folder
cd /d "%~dp0"

REM Create venv if it doesn't exist
if not exist "env\Scripts\activate.bat" (
    echo Creating Python virtual environment...
    python -m venv env
    call "env\Scripts\activate.bat"
    pip install -r requirements.txt
) else (
    call "env\Scripts\activate.bat"
)

REM Run your Python script
python "TrainV4.py" %*

REM Pause so you can see output if double-clicked
pause
