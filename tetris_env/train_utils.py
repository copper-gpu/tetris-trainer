from __future__ import annotations

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics

from .env import TetrisEnv


def make_training_envs(n_envs: int = 16):
    return DummyVecEnv([lambda: TetrisEnv() for _ in range(n_envs)])


def make_eval_env(max_episode_steps: int = 2000):
    return Monitor(
        TimeLimit(
            RecordEpisodeStatistics(TetrisEnv()),
            max_episode_steps
        )
    )
