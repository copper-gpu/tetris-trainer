"""
live_view.py
============

Continuously plays the latest checkpoints/best_model.zip in a Pygame window.
Once one episode finishes, it immediately starts the next—no “hang” after the first piece.

Key points:
- Keeps one persistent TetrisEnv + window open forever
- Loads model weights only when the file’s MD5 hash actually changes
- Inference runs on CPU so it never steals GPU memory from training
- Services Pygame events every loop so the window never freezes/“Not Responding”
- ESC or closing the window quits cleanly
"""

import time
import hashlib
import pygame
import sys
from pathlib import Path
from stable_baselines3 import PPO
from tetris_env import TetrisEnv

BEST_MODEL = Path("checkpoints/best_model.zip")
last_hash  = None       # MD5 of the last‐loaded checkpoint
model      = None
device     = "cpu"      # do inference on CPU to leave GPU for training

# ── Helper to compute MD5 on the checkpoint file ──────────────────
def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

# ── Initialize Pygame + one persistent TetrisEnv + window ─────────
pygame.init()
FONT = pygame.font.SysFont("consolas", 28, bold=True)

env = TetrisEnv()
env.render("human")   # opens Pygame window
env.reset()           # first reset

def overlay(text: str):
    """
    Draw a translucent overlay with `text` centered in the window.
    """
    surf = pygame.Surface(env.window.get_size(), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))  # semi‐transparent black
    txt  = FONT.render(text, True, (255, 255, 255))
    surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
    env.window.blit(surf, (0, 0))
    pygame.display.flip()

# ── Main loop ─────────────────────────────────────────────────────
while True:
    # 1) Service Pygame events to keep the window responsive
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
            pygame.quit()
            sys.exit()

    # 2) If a new checkpoint appears, load it
    if BEST_MODEL.exists():
        current_hash = md5(BEST_MODEL)
        if current_hash != last_hash:
            overlay("Loading best model …")
            try:
                model = PPO.load(BEST_MODEL, env=env, device=device)
                last_hash = current_hash
                print(f"🔄  Reloaded {BEST_MODEL}  (hash {current_hash[:8]})")
            except Exception as e:
                print("❌  Failed to load checkpoint:", e)
                time.sleep(1.0)
                continue
    else:
        overlay("Waiting for checkpoints/best_model.zip …")
        time.sleep(1.0)
        continue

    # 3) Play one episode, then loop back for the next
    try:
        obs, _ = env.reset()   # Always reset at the start of each new episode
        done = False
        while not done:
            # keep window responsive inside the episode
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()

            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
            time.sleep(0.02)   # about 50 FPS
        # ── Episode ended; loop returns to top, resets again ──

    except Exception as e:
        print("⚠️  Runtime error during play:", e, "– restarting episode")
        time.sleep(0.5)
        # Loop will naturally go back, reset, and try again
