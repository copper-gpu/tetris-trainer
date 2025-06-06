@echo off
REM Quickstart script for Tetris Trainer on Windows
REM Installs dependencies and launches the CLI menu

python -m pip install -e .
python -m pip install -r requirements.txt
python cli_menu.py
