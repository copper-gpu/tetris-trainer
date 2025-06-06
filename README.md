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
