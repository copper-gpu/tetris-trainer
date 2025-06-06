#!/bin/bash
# Quickstart script for Tetris Trainer
# Installs dependencies and launches the CLI menu
set -e

# install the local package in editable mode using the same Python
python3 -m pip install -e .

# install required runtime dependencies
python3 -m pip install -r requirements.txt

# start the menu with the same interpreter
python3 cli_menu.py
