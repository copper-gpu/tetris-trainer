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
import sys
from pathlib import Path
import threading

try:  # Ensure pygame is available
    import pygame
except ModuleNotFoundError as e:  # pragma: no cover - for runtime diagnostics
    print("Error: pygame is not installed. Run 'pip install -r requirements.txt' first.")
    raise SystemExit(1) from e

try:  # Ensure stable_baselines3 is installed
    from stable_baselines3 import PPO
except ModuleNotFoundError as e:  # pragma: no cover - for runtime diagnostics
    print("Error: stable-baselines3 is not installed. Run 'pip install -r requirements.txt' first.")
    raise SystemExit(1) from e

try:  # Ensure editable install of this repo
    from tetris_env import TetrisEnv
except ModuleNotFoundError as e:  # pragma: no cover - for runtime diagnostics
    print("Error: could not import tetris_env. Did you run 'pip install -e .'?")
    raise SystemExit(1) from e

BEST_MODEL = Path("checkpoints/best_model.zip")
last_hash  = None       # MD5 of the last‐loaded checkpoint
model      = None
device     = "cpu"      # do inference on CPU to leave GPU for training

# Async model loading helpers
load_thread      = None
load_exception   = None
loaded_model     = None
hash_being_loaded = None

# ── Helper to compute MD5 on the checkpoint file ──────────────────
def md5(path: Path) -> str | None:
    """Return the MD5 hash of ``path`` or ``None`` if the file disappears."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    except FileNotFoundError:  # file replaced while reading
        return None
    return h.hexdigest()

# ── Initialize Pygame + one persistent TetrisEnv + window ─────────
def main() -> None:
    pygame.init()
    global env, FONT
    global load_thread, load_exception, loaded_model, hash_being_loaded
    global model, last_hash

    FONT = pygame.font.SysFont("consolas", 28, bold=True)

    env = TetrisEnv()
    env.render("human")   # opens Pygame window
    env.reset()           # first reset

    def overlay(text: str):
        """
        Draw a translucent overlay with ``text`` centered in the window.
        """
        if env.renderer is None or env.renderer.window is None:
            return
        surf = pygame.Surface(env.renderer.window.get_size(), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 180))  # semi-transparent black
        txt = FONT.render(text, True, (255, 255, 255))
        surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
        env.renderer.window.blit(surf, (0, 0))
        pygame.display.flip()


    def _load_model_async():
        """Background thread target for loading the PPO checkpoint."""
        global loaded_model, load_exception
        try:
            # Load the PPO checkpoint without attaching the live environment.
            # stable_baselines3 will access the environment's spaces during load,
            # which is safe, but calling env.reset() inside this background thread
            # can freeze Pygame. By loading without ``env`` and assigning it on
            # the main thread, we avoid any Pygame calls outside the main loop.
            loaded_model = PPO.load(
                BEST_MODEL,
                device=device,
                custom_objects={"n_envs": 1},  # allow attaching single-env later
            )
        except Exception as e:
            load_exception = e

    # ── Main loop ─────────────────────────────────────────────────
    while True:
        # 1) Service Pygame events to keep the window responsive
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit()
                sys.exit()

        # 2) Load checkpoints asynchronously so the window never freezes
        if load_thread is not None:
            if load_thread.is_alive():
                overlay("Loading best model …")
                time.sleep(0.1)
                continue
            load_thread.join()
            if load_exception:
                print("❌  Failed to load checkpoint:", load_exception)
                load_exception = None
                load_thread = None
                time.sleep(1.0)
                continue
            model = loaded_model
            if model is not None:
                # Attach the live environment on the main thread where all
                # Pygame interactions happen. Calling set_env here ensures we
                # avoid any Pygame calls from the loader thread.
                model.set_env(env)
            loaded_model = None
            last_hash = hash_being_loaded
            print(f"🔄  Reloaded {BEST_MODEL}  (hash {last_hash[:8]})")
            load_thread = None
            continue

        if not BEST_MODEL.exists():
            overlay("Waiting for checkpoints/best_model.zip …")
            time.sleep(1.0)
            continue

        current_hash = md5(BEST_MODEL)
        if current_hash is None:
            time.sleep(0.1)
            continue
        if current_hash != last_hash:
            overlay("Loading best model …")
            hash_being_loaded = current_hash
            load_thread = threading.Thread(target=_load_model_async)
            load_thread.start()
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


if __name__ == "__main__":
    main()
