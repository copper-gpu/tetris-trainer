# resume_training.py

"""
Continue PPO training from 10M → 20M (or beyond).
Writes new logs under logs/run_02/ (or your chosen run_id),
appends to the same checkpoints/best_model.zip, and saves
the new final model as ppo_tetris_offline_20M.zip.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from gymnasium.wrappers import TimeLimit, RecordEpisodeStatistics
from tetris_env import TetrisEnv

# ── 1. Build training envs ─────────────────────────────────
N_ENVS = 16
env = DummyVecEnv([lambda: TetrisEnv() for _ in range(N_ENVS)])

# ── 2. Build evaluation env (monitor + cap + stats) ─────────
MAX_EVAL_LEN = 2_000  # cap each episode at 2K steps
eval_env = Monitor(
    TimeLimit(
        RecordEpisodeStatistics(TetrisEnv()),
        MAX_EVAL_LEN
    )
)

# ── 3. Load previous 10M-step checkpoint ────────────────────
#    Adjust the filename if you saved under a different name
model = PPO.load("ppo_tetris_offline_10M.zip", env=env, device="cuda")

# ── 4. New logger folder so curves don’t overlap ────────────
run_id = "run_02"
logger = configure(
    folder=f"logs/{run_id}",
    format_strings=["stdout", "tensorboard", "csv"]
)
model.set_logger(logger)

# ── 5. Evaluation callback (same settings as before) ────────
EVAL_FREQ = 32_768  # one full rollout (n_steps × n_envs)
callback = EvalCallback(
    eval_env,
    best_model_save_path="checkpoints",
    log_path=f"logs/{run_id}",
    eval_freq=EVAL_FREQ,
    deterministic=True,
    render=False
)

# ── 6. Resume training for another 10M steps ────────────────
model.learn(
    total_timesteps       = 10_000_000,
    callback              = callback,
    reset_num_timesteps   = False,   # keep global step counter
    log_interval          = 1        # print every PPO update
)

# ── 7. Save the new 20M-step checkpoint ──────────────────────
millions = model.num_timesteps // 1_000_000  # should now be 20
fname    = f"ppo_tetris_offline_{millions}M"
model.save(fname)
print(f"✅  Training resumed & saved to {fname}.zip")
