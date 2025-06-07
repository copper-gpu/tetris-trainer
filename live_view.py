"""View the latest ``checkpoints/best_model.zip`` in a Pygame window.

The script keeps a single ``TetrisEnv`` running and reloads the model whenever
``best_model.zip`` changes. Press <ESC> or close the window to quit.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import pygame
import torch
from stable_baselines3 import PPO

from tetris_env import TetrisEnv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL = Path("checkpoints/best_model.zip")


def file_hash(path: Path) -> str:
    """Return the MD5 hash of ``path``."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def handle_events() -> None:
    """Process Pygame events and exit on window close or ESC."""
    for ev in pygame.event.get():
        if ev.type == pygame.QUIT or (
            ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE
        ):
            pygame.quit()
            sys.exit()


def overlay(env: TetrisEnv, font: pygame.font.Font, text: str) -> None:
    """Draw a translucent overlay with ``text`` centred in the window."""
    if env.renderer is None or env.renderer.window is None:
        return
    surf = pygame.Surface(env.renderer.window.get_size(), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 180))
    txt = font.render(text, True, (255, 255, 255))
    surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
    env.renderer.window.blit(surf, (0, 0))
    pygame.display.flip()


def main() -> None:
    pygame.init()
    font = pygame.font.SysFont("consolas", 28, bold=True)

    env = TetrisEnv()
    env.render("human")
    env.reset()

    model = None
    last_hash: str | None = None

    while True:
        handle_events()

        if not BEST_MODEL.exists():
            overlay(env, font, "Waiting for checkpoints/best_model.zip …")
            time.sleep(1.0)
            continue

        current_hash = file_hash(BEST_MODEL)
        if model is None or current_hash != last_hash:
            overlay(env, font, "Loading best model …")
            model = PPO.load(BEST_MODEL, env=env, device=DEVICE)
            last_hash = current_hash
            env.reset()

        obs, _ = env.reset()
        done = False
        while not done:
            handle_events()
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
            time.sleep(0.02)


if __name__ == "__main__":
    main()
