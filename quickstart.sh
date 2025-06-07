#!/bin/bash
# Quickstart script for Tetris Trainer
# Creates a virtual environment, installs dependencies and launches the CLI menu
set -e

# create a virtual environment if one does not exist
if [ ! -d .venv ]; then
    python3 -m venv .venv
fi

# activate the environment
source .venv/bin/activate

# install the local package and runtime dependencies
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# start the menu with the environment's interpreter
python cli_menu.py
