from __future__ import annotations
import random
import numpy as np

from .constants import BOARD_WIDTH, BOARD_HEIGHT, BUFFER_ROWS, Action

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


class TetrisCore:
    WIDTH, HEIGHT, BUFFER = BOARD_WIDTH, BOARD_HEIGHT, BUFFER_ROWS

    def __init__(self, seed: int | None = None, preview: int = 5):
        self.random = random.Random(seed)
        self.preview_len = preview
        self.reset()

    # ── lifecycle ──
    def reset(self):
        """Initialize a fresh board and spawn first piece."""
        self.board = np.zeros((self.HEIGHT + self.BUFFER, self.WIDTH), dtype=np.int8)
        self.lines = 0
        self.hold_piece = None
        self.queue = []
        self._fill_queue()
        self._spawn(None)
        return self._obs(), {}

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

    def _hold(self):
        if self.hold_used:
            return
        self.hold_used = True
        self.current_piece, self.hold_piece = (
            self.hold_piece or self.queue.pop(0),
            self.current_piece,
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
            coords = [tuple((mat @ np.array(p)).tolist()) for p in coords]
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

    def _lock(self):
        for x, y in self._cells(self.pos, self.rot):
            if 0 <= y < self.board.shape[0]:
                self.board[y, x] = PIECE_IDS[self.current_piece]
        full = [i for i in range(self.board.shape[0]) if all(self.board[i])]
        n = len(full)
        if n:
            self.board = np.delete(self.board, full, axis=0)
            self.board = np.vstack([
                self.board,
                np.zeros((n, self.WIDTH), np.int8)
            ])
            self.lines += n
        self._spawn(None)
        return n * n * 10 + 4 * n

    # ──────────────────────────────────────────────────────────
    def step(self, action: int | Action):
        reward = 0.0
        done = False
        a = Action(action)

        prev_x = int(self.pos[0])
        locked = False

        if a == Action.LEFT:
            self._shift(-1)
        elif a == Action.RIGHT:
            self._shift(1)
        elif a == Action.ROT_CW:
            self._rotate(+1)
        elif a == Action.ROT_CCW:
            self._rotate(-1)
        elif a == Action.SOFT_DROP:
            if self._soft_drop():
                reward += 1.0
        elif a == Action.HARD_DROP:
            drop_cells = self._hard_drop()
            reward += drop_cells * 2.0
            line_reward = self._lock()
            reward += line_reward
            locked = True
        elif a == Action.HOLD:
            self._hold()

        if a not in (Action.SOFT_DROP, Action.HARD_DROP):
            if not self._soft_drop():
                line_reward = self._lock()
                reward += line_reward
                locked = True

        current_x = int(self.pos[0])
        reward += 0.05 * abs(current_x - prev_x)

        if locked:
            board_vis = self.board[:-self.BUFFER]
            filled = board_vis > 0
            filled_above = np.maximum.accumulate(filled[::-1], axis=0)[::-1]
            holes = np.sum((board_vis == 0) & filled_above)
            reward -= float(holes)

        if np.any(self.board[-self.BUFFER:]):
            done = True

        return self._obs(), reward, done, False, {}

    def _obs(self):
        return {
            "board":   self.board[:-self.BUFFER].astype(np.float32, copy=False),
            "current": PIECE_IDS[self.current_piece],
            "pos":     self.pos.copy(),
            "rot":     self.rot,
            "hold":    PIECE_IDS.get(self.hold_piece, 0),
            "next":    np.array([PIECE_IDS[p] for p in self.queue[:5]], np.int8),
            "lines":   self.lines,
        }
