# Tetris Trainer

This repository provides a small Gym-compatible Tetris environment used for reinforcement learning experiments.

## Quickstart

Run the quickstart script to install all dependencies and launch the
menu-driven interface in one step. On Linux or macOS use `quickstart.sh`
(which relies on `python3` being on your `PATH`). On Windows run the new
`quickstart.bat` file.

```bash
./quickstart.sh      # Linux/macOS
```

```bat
quickstart.bat       # Windows
```

Both scripts install the package in editable mode, install the runtime
dependencies from `requirements.txt` and then start `cli_menu.py`.

## Installation

Install the package in editable mode so the `tetris_env` module can be imported by tests and example scripts:

```bash
pip install -e .
```

Then install all runtime dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

After installation you can run the example scripts in this repository or your own code that imports `tetris_env`.

## Running tests

Install the package and runtime dependencies first:

```bash
pip install -e .
pip install -r requirements.txt
```

Then run `pytest` to execute the automated checks in the `tests` directory.
These tests create a `TetrisEnv` instance and verify core behaviours including
environment resets, piece movement, hard drop locking and game over detection.

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
4. View the latest `checkpoints/best_model.zip` with `live_view.py`.
5. Exit the program.

Each option launches the corresponding script so you do not need to
remember individual commands.

When choosing options 2 or 3 you will be prompted for how many
environment steps to run. Press <kbd>Enter</kbd> to keep the default
10,000,000. Both training modes automatically increment the log folder
(``logs/run_01``, ``logs/run_02`` …) so each run is kept separate. When
resuming training, the script also asks which existing checkpoint to
load before continuing.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
