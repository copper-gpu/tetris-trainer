"""Simple manual-play demo for the Tetris environment."""
import sys

try:
    import pygame
except ImportError:
    sys.exit("Install pygame:  pip install pygame")

from tetris_env import TetrisEnv, Action

KEYMAP = {
    pygame.K_LEFT: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_z: Action.ROT_CCW,
    pygame.K_x: Action.ROT_CW,
    pygame.K_UP: Action.ROT_CW,
    pygame.K_DOWN: Action.SOFT_DROP,
    pygame.K_SPACE: Action.HARD_DROP,
    pygame.K_c: Action.HOLD,
}

def manual_demo():
    env = TetrisEnv()
    env.render("human")
    obs, _ = env.reset()
    done = False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r and done:
                    obs, _ = env.reset()
                    done = False
                elif ev.key in KEYMAP and not done:
                    obs, _, done, _, _ = env.step(KEYMAP[ev.key])
        if not done:
            obs, _, done, _, _ = env.step(Action.NONE)
        if done:
            surf = pygame.Surface(env.renderer.window.get_size(), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 180))
            font = pygame.font.SysFont("consolas", 48, bold=True)
            txt = font.render("GAME OVER", True, (255, 80, 80))
            surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
            font2 = pygame.font.SysFont("consolas", 22)
            hint = font2.render("Press R to restart", True, (230, 230, 230))
            surf.blit(hint, hint.get_rect(center=(surf.get_width()//2, surf.get_height()//2 + 60)))
            env.renderer.window.blit(surf, (0, 0))
            pygame.display.flip()
        pygame.time.wait(50)

if __name__ == "__main__":
    manual_demo()
