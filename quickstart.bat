@echo off
REM Quickstart script for Tetris Trainer on Windows
REM Creates a virtual environment, installs dependencies and launches the CLI menu

IF NOT EXIST .venv (
    python -m venv .venv
)
call .venv\Scripts\activate.bat

python -m pip install -e .
python -m pip install -r requirements.txt
python cli_menu.py
