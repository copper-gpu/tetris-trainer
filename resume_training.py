# resume_training.py

"""
resume_training.py
==================

Interactive script to continue PPO training from a previous checkpoint.
It automatically creates a new log folder (``logs/run_XX``) for every
resume invocation and allows selecting which ``ppo_tetris_offline_*M.zip``
checkpoint to load. This avoids manual edits when you want to roll back to
an earlier model.

The script keeps writing checkpoints/best_model.zip and saves the new final
model as ``ppo_tetris_offline_<N>M.zip`` where ``N`` is the accumulated number
of millions of environment steps.
"""

import argparse
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from tetris_env.train_utils import make_training_envs, make_eval_env


def next_run_id() -> str:
    """Return ``run_XX`` where ``XX`` is one higher than existing log folders."""
    logs = Path("logs")
    logs.mkdir(exist_ok=True)
    numbers = []
    for d in logs.iterdir():
        m = re.match(r"run_(\d+)", d.name)
        if m:
            numbers.append(int(m.group(1)))
    next_num = max(numbers, default=1) + 1
    return f"run_{next_num:02d}"


def select_checkpoint(specified: str | None) -> Path:
    """Find and optionally prompt for a checkpoint to load."""
    if specified:
        path = Path(specified)
        if not path.exists():
            raise SystemExit(f"Checkpoint {path} not found")
        return path

    ckpts = sorted(
        Path(".").glob("ppo_tetris_offline_*M.zip"),
        key=lambda p: int(re.search(r"(\d+)M", p.stem).group(1)),
    )
    if not ckpts:
        raise SystemExit("No checkpoints found. Run train_offline.py first.")

    print("Available checkpoints:")
    for i, ck in enumerate(ckpts, 1):
        print(f" {i}. {ck.name}")
    choice = input(f"Select checkpoint [default {ckpts[-1].name}]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(ckpts):
        return ckpts[int(choice) - 1]
    return ckpts[-1]

def main() -> None:
    parser = argparse.ArgumentParser(description="Resume PPO training")
    parser.add_argument(
        "-c",
        "--checkpoint",
        help="Checkpoint file to load (default: choose interactively)",
    )
    args = parser.parse_args()

    # ── 1. Build training and eval envs ─────────────────────
    N_ENVS = 16
    MAX_EVAL_LEN = 2_000
    env = make_training_envs(N_ENVS)
    eval_env = make_eval_env(MAX_EVAL_LEN)

    # ── 2. Select and load checkpoint ───────────────────────
    ckpt_path = select_checkpoint(args.checkpoint)
    model = PPO.load(ckpt_path, env=env, device="cuda")

    # ── 3. Logger (auto-increment run id) ───────────────────
    run_id = next_run_id()
    logger = configure(
        folder=f"logs/{run_id}",
        format_strings=["stdout", "tensorboard", "csv"],
    )
    model.set_logger(logger)

    # ── 4. Evaluation callback ──────────────────────────────
    EVAL_FREQ = 32_768  # one full rollout (n_steps × n_envs)
    callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",
        log_path=f"logs/{run_id}",
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
    )

    # ── 5. Resume training for another 10M steps ────────────
    model.learn(
        total_timesteps=10_000_000,
        callback=callback,
        reset_num_timesteps=False,
        log_interval=1,
    )

    # ── 6. Save checkpoint with updated step count ─────────
    millions = model.num_timesteps // 1_000_000
    fname = f"ppo_tetris_offline_{millions}M"
    model.save(fname)
    print(f"✅  Training resumed & saved to {fname}.zip")


if __name__ == "__main__":
    main()
