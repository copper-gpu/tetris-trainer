# Tetris Trainer

This repository provides a small Gym-compatible Tetris environment used for reinforcement learning experiments.

## Installation

Install the package in editable mode so the `tetris_env` module can be imported by tests and example scripts:

```bash
pip install -e .
```

The environment requires NumPy and either Gymnasium or Gym. If they are not already installed, run:

```bash
pip install numpy gymnasium gym
```

After installation you can run the example scripts in this repository or your own code that imports `tetris_env`.

## Running tests

```bash
pytest
```

## Command Line Interface

A simple menu-driven interface is provided in `cli_menu.py` to make common
actions easy for beginners. Run the script and pick an option:

```bash
python cli_menu.py
```

The menu lets you:

1. Play Tetris manually using the `play.py` demo.
2. Start training from scratch via `train_offline.py`.
3. Resume training from a saved checkpoint with `resume_training.py`.
4. Exit the program.

Each option launches the corresponding script so you do not need to
remember individual commands.

When resuming training, the script now asks which existing checkpoint to
load and automatically increments the log folder (``logs/run_02``,
``logs/run_03`` …) so your progress is clearly separated.
