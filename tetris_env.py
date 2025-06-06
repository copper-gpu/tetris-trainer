"""
tetris_env.py
=============

Modern Guideline-style Tetris with a Gym-compatible interface and Pygame rendering.

* 10 × 20 matrix, 4 hidden buffer rows
* 7-bag RNG, Hold, 5-piece preview
* SRS rotation kicks
* Gym-compatible: `env = TetrisEnv()`
* Optional Pygame renderer for human play
* Python 3.8+ (no walrus operators)
"""

from __future__ import annotations
import random, sys
import numpy as np

# Gymnasium preferred; fallback to gym
try:
    import gymnasium as gym
except ImportError:
    import gym

# ────────────────── SRS CONSTANTS ──────────────────
ROT_RIGHT = np.array([[0, -1], [1, 0]])
ROT_LEFT  = np.array([[0,  1], [-1, 0]])

PIECES = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}
PIECE_IDS = {k: i + 1 for i, k in enumerate("IOTSZJL")}  # 1–7

JLTSZ_KICKS = {
    (0, 1): [(0, 0), (-1, 0), (-1, 1), ( 0, -2), (-1, -2)],
    (1, 0): [(0, 0), ( 1, 0), ( 1, -1), ( 0,  2), ( 1,  2)],
    (1, 2): [(0, 0), ( 1, 0), ( 1, -1), ( 0,  2), ( 1,  2)],
    (2, 1): [(0, 0), (-1, 0), (-1, 1), ( 0, -2), (-1, -2)],
    (2, 3): [(0, 0), ( 1, 0), ( 1,  1), ( 0, -2), ( 1, -2)],
    (3, 2): [(0, 0), (-1, 0), (-1, -1), ( 0,  2), (-1,  2)],
    (3, 0): [(0, 0), (-1, 0), (-1, -1), ( 0,  2), (-1,  2)],
    (0, 3): [(0, 0), ( 1, 0), ( 1,  1), ( 0, -2), ( 1, -2)],
}
I_KICKS = {
    (0, 1): [(0, 0), (-2, 0), ( 1, 0), (-2, -1), ( 1,  2)],
    (1, 0): [(0, 0), ( 2, 0), (-1, 0), ( 2,  1), (-1, -2)],
    (1, 2): [(0, 0), (-1, 0), ( 2, 0), (-1,  2), ( 2, -1)],
    (2, 1): [(0, 0), ( 1, 0), (-2, 0), ( 1, -2), (-2,  1)],
    (2, 3): [(0, 0), ( 2, 0), (-1, 0), ( 2,  1), (-1, -2)],
    (3, 2): [(0, 0), (-2, 0), ( 1, 0), (-2, -1), ( 1,  2)],
    (3, 0): [(0, 0), ( 1, 0), (-2, 0), ( 1, -2), (-2,  1)],
    (0, 3): [(0, 0), (-1, 0), ( 2, 0), (-1,  2), ( 2, -1)],
}

def get_kicks(piece: str, fr: int, to: int):
    if piece == "O":
        return [(0, 0)]
    if piece == "I":
        return I_KICKS[(fr % 4, to % 4)]
    return JLTSZ_KICKS[(fr % 4, to % 4)]


# ────────────────── CORE GAME LOGIC ──────────────────
class TetrisCore:
    WIDTH, HEIGHT, BUFFER = 10, 20, 4  # buffer rows at top
    ACTIONS = {
        0: "left", 1: "right", 2: "rotCW", 3: "rotCCW",
        4: "softDrop", 5: "hardDrop", 6: "hold", 7: "none"
    }

    def __init__(self, seed: int | None = None, preview: int = 5):
        self.random = random.Random(seed)
        self.preview_len = preview
        self.reset()

    # ── lifecycle ──
    def reset(self):
        """
        Initialize a fresh empty board, zero lines, clear hold, refill queue, and spawn first piece.
        Returns:
            obs (dict), info (empty dict)
        """
        # Empty board: height + buffer rows
        self.board = np.zeros((self.HEIGHT + self.BUFFER, self.WIDTH), dtype=np.int8)
        self.lines = 0
        self.hold_piece = None
        self.queue = []
        self._fill_queue()
        self._spawn(None)
        return self._obs(), {}

    # bag / queue
    def _fill_queue(self):
        while len(self.queue) < self.preview_len:
            bag = list("IOTSZJL")
            self.random.shuffle(bag)
            self.queue.extend(bag)

    def _spawn(self, piece: str | None = None):
        self.current_piece = piece or self.queue.pop(0)
        self._fill_queue()
        self.pos, self.rot = np.array([3, self.HEIGHT + 1]), 0
        self.hold_used = False

    # hold
    def _hold(self):
        if self.hold_used:
            return
        self.hold_used = True
        self.current_piece, self.hold_piece = (
            self.hold_piece or self.queue.pop(0),
            self.current_piece
        )
        self._fill_queue()
        self.pos, self.rot = np.array([3, self.HEIGHT + 1]), 0

    # ── geometry helpers ──
    def _cells(self, pos, rot, piece=None):
        piece = piece or self.current_piece
        coords = PIECES[piece]
        r = rot % 4
        if r:
            mat = {1: ROT_RIGHT, 2: ROT_RIGHT @ ROT_RIGHT, 3: ROT_LEFT}[r]
            coords = [
                tuple((mat @ np.array(p)).tolist())
                for p in coords
            ]
        for x, y in coords:
            yield pos[0] + x, pos[1] - y

    def _collision(self, pos, rot):
        for x, y in self._cells(pos, rot):
            if x < 0 or x >= self.WIDTH or y < 0:
                return True
            if y >= self.board.shape[0]:
                continue
            if self.board[y, x]:
                return True
        return False

    # ── movement ──
    def _shift(self, dx: int):
        np_pos = self.pos + np.array([dx, 0])
        if not self._collision(np_pos, self.rot):
            self.pos = np_pos

    def _rotate(self, d: int):
        nr = (self.rot + d) % 4
        for ox, oy in get_kicks(self.current_piece, self.rot, nr):
            np_pos = self.pos + np.array([ox, oy])
            if not self._collision(np_pos, nr):
                self.pos, self.rot = np_pos, nr
                break

    def _soft_drop(self):
        np_pos = self.pos + np.array([0, -1])
        if self._collision(np_pos, self.rot):
            return False
        self.pos = np_pos
        return True

    def _hard_drop(self):
        d = 0
        while self._soft_drop():
            d += 1
        return d

    # lock / clear
    def _lock(self):
        for x, y in self._cells(self.pos, self.rot):
            if 0 <= y < self.board.shape[0]:
                self.board[y, x] = PIECE_IDS[self.current_piece]
        full = [i for i in range(self.board.shape[0]) if all(self.board[i])]
        n = len(full)
        if n:
            self.board = np.delete(self.board, full, axis=0)
            # New empty rows should appear at the top of the playfield
            # (higher indices) since row 0 represents the bottom.
            self.board = np.vstack([
                self.board,
                np.zeros((n, self.WIDTH), np.int8)
            ])
            self.lines += n
        self._spawn(None)
        return n * n * 10 + 4 * n

    # ──────────────────────────────────────────────────────────
    def step(self, action: int):
        """
        One game tick with extra shaping:
          • +1.0 for each cell soft-dropped (downward move)
          • +0.05 × horizontal distance moved this tick
          • Line‐clear rewards as before (n²×10 + 4n)
          • −1.0 × number of holes after lock (holes = empty cells with blocks above)
        """
        reward = 0.0
        done = False
        a = self.ACTIONS.get(action, "none")

        # ── remember x before the action for horizontal bonus ──
        prev_x = int(self.pos[0])

        # ── execute the chosen action (track if we locked) ─────
        locked = False

        if a == "left":
            self._shift(-1)

        elif a == "right":
            self._shift(1)

        elif a == "rotCW":
            self._rotate(+1)

        elif a == "rotCCW":
            self._rotate(-1)

        elif a == "softDrop":
            if self._soft_drop():
                reward += 1.0  # stronger soft‐drop bonus

        elif a == "hardDrop":
            drop_cells = self._hard_drop()
            reward += drop_cells * 2.0
            # locking after a hard drop always triggers _lock()
            line_reward = self._lock()
            reward += line_reward
            locked = True

        elif a == "hold":
            self._hold()

        # ── automatic gravity tick if we didn’t already hard-drop ──
        if a not in ("softDrop", "hardDrop"):
            if not self._soft_drop():
                # piece can no longer fall → lock into place
                line_reward = self._lock()
                reward += line_reward
                locked = True

        # ── horizontal‐movement bonus (per cell moved) ───────────
        current_x = int(self.pos[0])
        reward += 0.05 * abs(current_x - prev_x)

        # ── hole penalty if we just locked a piece ──────────────
        if locked:
            # Visible board (exclude bottom BUFFER rows)
            board_vis = self.board[:-self.BUFFER]
            holes = 0
            # Count any empty cell (0) that has at least one block above it
            for col in range(self.WIDTH):
                col_data = board_vis[:, col]
                # For each row from top (0) to second-last (HEIGHT-2):
                for row in range(self.HEIGHT - 1):
                    if col_data[row] == 0 and np.any(col_data[row + 1 :] > 0):
                        holes += 1
            reward -= 1.0 * holes

        # ── game-over check ─────────────────────────────────────
        if np.any(self.board[-self.BUFFER :]):
            done = True

        # ── return the observation, reward, done, truncated, info ──
        return self._obs(), reward, done, False, {}

    # observation
    def _obs(self):
        return {
            "board":   self.board[:-self.BUFFER].astype(np.float32, copy=False),
            "current": PIECE_IDS[self.current_piece],
            "pos":     self.pos.copy(),
            "rot":     self.rot,
            "hold":    PIECE_IDS.get(self.hold_piece, 0),
            "next":    np.array([PIECE_IDS[p] for p in self.queue[:5]], np.int8),
            "lines":   self.lines
        }


# ────────────────── GYM WRAPPER (with Pygame) ──────────────────
class TetrisEnv(TetrisCore, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, seed: int | None = None):
        TetrisCore.__init__(self, seed)
        # Default cell size so rgb_array rendering works before any window
        # has been created via _setup_pygame().
        self.CELL = 24
        self.action_space = gym.spaces.Discrete(len(self.ACTIONS))
        self.observation_space = gym.spaces.Dict({
            "board":   gym.spaces.Box(0, 7, (20, 10), np.float32),
            "current": gym.spaces.Discrete(8),
            "pos":     gym.spaces.Box(0, 20, (2,), np.int8),
            "rot":     gym.spaces.Discrete(4),
            "hold":    gym.spaces.Discrete(8),
            "next":    gym.spaces.Box(0, 7, (5,), np.int8),
            "lines":   gym.spaces.Discrete(10000)
        })
        # Keep track of whether a window has been created yet:
        self.render_mode = None
        self.window      = None
        self.clock       = None

    # ── Reset ─────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options=None):
        """
        Reset the Tetris environment.
        Only calls _setup_pygame() the first time a human render is requested;
        on subsequent resets it simply redraws the existing window.
        Returns:
            obs (dict), info (empty dict)
        """
        if seed is not None:
            self.random.seed(seed)

        # Call TetrisCore.reset() to reinitialize game state and get obs
        obs, info = super().reset()

        # If we’re in human-render mode, ensure window exists and draw once:
        if self.render_mode == "human":
            import pygame
            if self.window is None:
                self._setup_pygame()
            self._draw()

        return obs, info

    # ── Step ──────────────────────────────────────────────────────
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        if self.render_mode == "human":
            self._draw()
            self.clock.tick(self.metadata["render_fps"])
        return obs, reward, done, False, info

    # ── Render ───────────────────────────────────────────────────
    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "human":
            import pygame
            if self.window is None:
                self._setup_pygame()
            self._draw()
        elif mode == "rgb_array":
            return self._array_frame()

    # ── Close ────────────────────────────────────────────────────
    def close(self):
        if self.window:
            import pygame
            pygame.quit()
            self.window = None

    # ── pygame helpers ──
    def _setup_pygame(self):
        """
        Initialize Pygame and open a window large enough for:
        • 120-px HOLD box  (5×24)
        • 20-px gutter
        • 240-px board     (10×24)
        • 20-px gutter
        • 120-px NEXT box  (5×24)
        → 120 + 20 + 240 + 20 + 120 = 520
        We add extra margin and round to 640 × 600.
        """
        import pygame
        pygame.init()
        self.CELL = 24
        WIN_W, WIN_H = 640, 600   # wide enough so nothing gets clipped
        self.window = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

    def _array_frame(self):
        import pygame
        surf = pygame.Surface((10 * self.CELL, 20 * self.CELL))
        self._draw_board(surf)
        return pygame.surfarray.array3d(surf)

    # ── Drawing routines ──
    COLORS = [(0, 0, 0), (0, 255, 255), (255, 255, 0), (128, 0, 128),
              (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 128, 0)]

    def _draw(self):
        import pygame
        self.window.fill((15, 17, 26))

        # Board at (180,40)
        board_surf = pygame.Surface((10 * self.CELL, 20 * self.CELL))
        self._draw_board(board_surf, blit_to=self.window, origin=(180, 40))

        # HOLD box at (20,40)
        self._draw_piece_box(self.hold_piece, (20, 40), "HOLD")

        # NEXT box now starts at x = 460  (board right edge 420 + 40-px gutter)
        self._draw_next(origin=(460, 40))

        # Lines counter
        font = pygame.font.SysFont("consolas", 18)
        self.window.blit(font.render(f"Lines: {self.lines}", True, (220, 220, 220)),
                         (20, 560))

        pygame.display.flip()

    def _draw_board(self, surf, blit_to=None, origin=(0, 0)):
        import pygame
        CELL = self.CELL
        surf.fill((0, 0, 0))
        # draw grid lines
        for x in range(11):
            pygame.draw.line(surf, (40, 40, 40), (x*CELL, 0), (x*CELL, 20*CELL))
        for y in range(21):
            pygame.draw.line(surf, (40, 40, 40), (0, y*CELL), (10*CELL, y*CELL))
        # draw placed blocks
        for row in range(20):
            for col in range(10):
                v = self.board[row, col]
                if v:
                    pygame.draw.rect(surf, self.COLORS[v],
                                     (col*CELL+1, (19-row)*CELL+1, CELL-2, CELL-2))
        # draw current falling piece
        for x, y in self._cells(self.pos, self.rot):
            if 0 <= y < 20:
                pygame.draw.rect(surf, self.COLORS[PIECE_IDS[self.current_piece]],
                                 (x*CELL+1, (19-y)*CELL+1, CELL-2, CELL-2))
        if blit_to:
            blit_to.blit(surf, origin)

    def _draw_piece_box(self, piece, topleft, label):
        import pygame
        CELL = self.CELL
        pygame.draw.rect(self.window, (255, 255, 255),
                         (*topleft, 5*CELL, 4*CELL), 2)
        font = pygame.font.SysFont("consolas", 18, True)
        self.window.blit(font.render(label, True, (255, 255, 255)),
                         (topleft[0] + 4, topleft[1] - 22))
        if piece:
            self._draw_mini(piece, (topleft[0] + CELL, topleft[1] + CELL))

    def _draw_next(self, origin=(460, 40)):
        """
        Draw the 5-piece NEXT queue.
        Default origin moved to x=460 so it never overlaps the board.
        """
        import pygame
        CELL = self.CELL
        pygame.draw.rect(self.window, (255, 255, 255),
                         (*origin, 5*CELL, 20*CELL), 2)
        font = pygame.font.SysFont("consolas", 18, bold=True)
        self.window.blit(font.render("NEXT", True, (255, 255, 255)),
                         (origin[0] + 4, origin[1] - 22))
        for i, p in enumerate(self.queue[:5]):
            self._draw_mini(p, (origin[0] + CELL, origin[1] + i*4*CELL))

    def _draw_mini(self, piece, pos):
        import pygame
        MINI = {
            "I": [(0,1),(1,1),(2,1),(3,1)],
            "O": [(0,0),(1,0),(0,1),(1,1)],
            "T": [(1,0),(0,1),(1,1),(2,1)],
            "S": [(1,0),(2,0),(0,1),(1,1)],
            "Z": [(0,0),(1,0),(1,1),(2,1)],
            "J": [(0,0),(0,1),(1,1),(2,1)],
            "L": [(2,0),(0,1),(1,1),(2,1)]
        }
        CELL = self.CELL // 2
        for x, y in MINI[piece]:
            pygame.draw.rect(self.window,
                             self.COLORS[PIECE_IDS[piece]],
                             (pos[0] + x*CELL, pos[1] + y*CELL, CELL, CELL))


# ────────────────── MANUAL DEMO ──────────────────
def manual_demo():
    try:
        import pygame
    except ImportError:
        sys.exit("Install pygame:  pip install pygame")

    env = TetrisEnv()
    env.render("human")
    obs, _ = env.reset()
    KEYMAP = {
        pygame.K_LEFT:  0, pygame.K_RIGHT: 1,
        pygame.K_z:     3, pygame.K_x:    2, pygame.K_UP: 2,
        pygame.K_DOWN:  4, pygame.K_SPACE: 5, pygame.K_c:  6
    }

    done = False
    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_r and done:      # restart
                    obs, _ = env.reset()
                    done = False
                elif ev.key in KEYMAP and not done:    # game input
                    obs, _, done, _, _ = env.step(KEYMAP[ev.key])

        if not done:                                   # gravity tick
            obs, _, done, _, _ = env.step(7)

        # If topped-out, overlay “Game Over”
        if done:
            import pygame
            surf = pygame.Surface(env.window.get_size(), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 180))                     # translucent black
            font = pygame.font.SysFont("consolas", 48, bold=True)
            txt  = font.render("GAME OVER", True, (255, 80, 80))
            surf.blit(txt, txt.get_rect(center=surf.get_rect().center))
            font2 = pygame.font.SysFont("consolas", 22)
            hint  = font2.render("Press R to restart", True, (230, 230, 230))
            surf.blit(hint, hint.get_rect(center=(
                surf.get_width()//2, surf.get_height()//2 + 60)))
            env.window.blit(surf, (0, 0))
            import pygame.display
            pygame.display.flip()

        import time
        time.sleep(0.05)


if __name__ == "__main__":
    manual_demo()
