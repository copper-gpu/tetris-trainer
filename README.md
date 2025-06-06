# Tetris Trainer

This repository contains a reinforcement learning environment and training scripts
for a modern Guideline-style Tetris game. It uses Stable Baselines3 with PPO and
provides utilities for offline training, resuming training, and viewing the
latest model in a Pygame window.

## Installation

Install Python dependencies with pip:

```bash
pip install -r requirements.txt
```

This installs `numpy`, `gymnasium`/`gym`, `pygame`, `stable-baselines3`, `torch`, and
any other packages required to run the code and tests.

## Usage

See `train_offline.py`, `resume_training.py`, and `live_view.py` for examples of
how to train and interact with the environment.
