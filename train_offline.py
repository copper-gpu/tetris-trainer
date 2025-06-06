"""
train_offline.py  –  PPO trainer with safe evaluation (fresh run)
================================================================

• 16 parallel envs, 2 048 steps/rollout  → 32 768 frames/update
• Mini-batch 8 192  (4 per update)
• Linear LR decay 5e-4 → 1e-5
• Evaluation every full rollout, each episode capped at 2 000 steps
• Console + TensorBoard + CSV logs under logs/run_01/
• Best model auto-saved to checkpoints/best_model.zip
• Final checkpoint saved as ppo_tetris_offline_<N>M.zip
"""

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn
from tetris_env.train_utils import make_training_envs, make_eval_env

# ── command line args ─────────────────────────────────────
parser = argparse.ArgumentParser(description="Train PPO offline")
parser.add_argument(
    "--steps",
    type=int,
    default=10_000_000,
    help="Total timesteps to train (default: 10_000_000)",
)
args = parser.parse_args()

# ── hyper-parameters ────────────────────────────────────
N_ENVS        = 16
ROLLOUT_STEPS = 2_048           # → 32 768 frames/update
BATCH_SIZE    = 8_192           # must evenly divide 32 768
TOTAL_STEPS   = args.steps      # total timesteps to train
MAX_EVAL_LEN  = 2_000           # cap each eval episode at 2 000 steps
EVAL_EVERY    = 32_768          # evaluate once per full rollout

# ── 1. Build training envs ───────────────────────────────
env = make_training_envs(N_ENVS)

# ── 2. Build evaluation env (Monitor + TimeLimit + stats) ─
eval_env = make_eval_env(MAX_EVAL_LEN)

# ── 3. Logger setup (write to logs/run_01/) ───────────────
RUN_ID = "run_01"
logger = configure(
    folder        = f"logs/{RUN_ID}",
    format_strings= ["stdout", "tensorboard", "csv"]
)

# ── 4. Evaluation callback (saves best model to checkpoints/) ─
callback = EvalCallback(
    eval_env,
    best_model_save_path = "checkpoints",
    log_path             = f"logs/{RUN_ID}",
    eval_freq            = EVAL_EVERY,
    deterministic        = True,
    render               = False
)

# ── 5. Create PPO model (new, from scratch) ─────────────────
model = PPO(
    "MultiInputPolicy",
    env,
    n_steps         = ROLLOUT_STEPS,
    batch_size      = BATCH_SIZE,
    learning_rate   = get_linear_fn(5e-4, 1e-5, TOTAL_STEPS),
    gamma           = 0.995,
    gae_lambda      = 0.95,
    clip_range      = 0.15,
    vf_coef         = 0.2,
    ent_coef        = 0.01,
    policy_kwargs   = dict(net_arch=[256,256,128]),
    device          = "cuda",      # or "cpu" if you don’t have a GPU
    verbose         = 1,           # console prints every update
    tensorboard_log = f"logs/{RUN_ID}"
)
model.set_logger(logger)

# ── 6. Train ────────────────────────────────────────────────
model.learn(
    total_timesteps        = TOTAL_STEPS,
    callback               = callback,
    reset_num_timesteps    = True,   # start from step 0
    log_interval           = 1       # print every PPO update
)

# ── 7. Save the final checkpoint ───────────────────────────
millions = model.num_timesteps // 1_000_000
fname    = f"ppo_tetris_offline_{millions}M"
model.save(fname)
print(f"✅  Training finished – saved checkpoint to {fname}.zip")
print("   Best model is in checkpoints/best_model.zip")
