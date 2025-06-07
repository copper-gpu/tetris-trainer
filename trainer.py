"""Unified entry point for training PPO on the Tetris environment.

This module consolidates ``train_offline.py`` and ``resume_training.py``.
Use the ``start`` subcommand to begin a new run and ``resume`` to continue
from an existing checkpoint.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import get_linear_fn

from tetris_env.train_utils import make_eval_env, make_training_envs


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

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
        raise SystemExit("No checkpoints found. Run start first.")

    print("Available checkpoints:")
    for i, ck in enumerate(ckpts, 1):
        print(f" {i}. {ck.name}")
    choice = input(f"Select checkpoint [default {ckpts[-1].name}]: ").strip()
    if choice.isdigit() and 1 <= int(choice) <= len(ckpts):
        return ckpts[int(choice) - 1]
    return ckpts[-1]


# ─────────────────────────────────────────────────────────────────────────────
# command implementations
# ─────────────────────────────────────────────────────────────────────────────

N_ENVS = 16
ROLLOUT_STEPS = 2_048          # → 32 768 frames/update
BATCH_SIZE = 8_192             # must evenly divide 32 768
MAX_EVAL_LEN = 2_000
EVAL_EVERY = 32_768            # evaluate once per full rollout


def start_cmd(args: argparse.Namespace) -> None:
    """Start a new training run, optionally from ``args.checkpoint``."""
    env = make_training_envs(N_ENVS)
    eval_env = make_eval_env(MAX_EVAL_LEN)

    run_id = next_run_id()
    logger = configure(folder=f"logs/{run_id}", format_strings=["stdout", "tensorboard", "csv"])

    callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",
        log_path=f"logs/{run_id}",
        eval_freq=EVAL_EVERY,
        deterministic=True,
        render=False,
    )

    if args.checkpoint:
        model = PPO.load(args.checkpoint, env=env, device="cuda")
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
            learning_rate=get_linear_fn(5e-4, 1e-5, args.steps),
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.15,
            vf_coef=0.2,
            ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256, 128]),
            device="cuda",
            verbose=1,
            tensorboard_log=f"logs/{run_id}",
        )

    model.set_logger(logger)
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        reset_num_timesteps=not args.checkpoint,
        log_interval=1,
    )

    millions = model.num_timesteps // 1_000_000
    fname = f"ppo_tetris_offline_{millions}M"
    model.save(fname)
    print(f"✅  Training finished – saved checkpoint to {fname}.zip")
    print("   Best model is in checkpoints/best_model.zip")


def resume_cmd(args: argparse.Namespace) -> None:
    """Resume training from a checkpoint."""
    env = make_training_envs(N_ENVS)
    eval_env = make_eval_env(MAX_EVAL_LEN)

    ckpt_path = select_checkpoint(args.checkpoint)
    model = PPO.load(ckpt_path, env=env, device="cuda")

    run_id = next_run_id()
    logger = configure(folder=f"logs/{run_id}", format_strings=["stdout", "tensorboard", "csv"])
    model.set_logger(logger)

    callback = EvalCallback(
        eval_env,
        best_model_save_path="checkpoints",
        log_path=f"logs/{run_id}",
        eval_freq=EVAL_EVERY,
        deterministic=True,
        render=False,
    )

    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        reset_num_timesteps=False,
        log_interval=1,
    )

    millions = model.num_timesteps // 1_000_000
    fname = f"ppo_tetris_offline_{millions}M"
    model.save(fname)
    print(f"✅  Training resumed & saved to {fname}.zip")


# ─────────────────────────────────────────────────────────────────────────────
# CLI setup
# ─────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Unified PPO trainer")
subparsers = parser.add_subparsers(dest="command", required=True)

start_parser = subparsers.add_parser("start", help="Start a new training run")
start_parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Total timesteps to train (default: 10_000_000)",
)
start_parser.add_argument(
    "-c",
    "--checkpoint",
    help="Optional checkpoint to initialize from",
)
start_parser.set_defaults(func=start_cmd)

resume_parser = subparsers.add_parser("resume", help="Resume from a checkpoint")
resume_parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Additional timesteps to train (default: 10_000_000)",
)
resume_parser.add_argument(
    "-c",
    "--checkpoint",
    help="Checkpoint file to load (default: choose interactively)",
)
resume_parser.set_defaults(func=resume_cmd)


if __name__ == "__main__":
    args = parser.parse_args()
    args.func(args)
