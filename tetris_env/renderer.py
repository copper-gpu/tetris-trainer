from __future__ import annotations

import pygame
import numpy as np

from .core import TetrisCore, PIECE_IDS

class PygameRenderer:
    COLORS = [
        (0, 0, 0), (0, 255, 255), (255, 255, 0), (128, 0, 128),
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 128, 0)
    ]

    def __init__(self, env: TetrisCore):
        self.env = env
        self.window = None
        self.clock = None
        self.CELL = 24

    def setup(self):
        pygame.init()
        self.CELL = 24
        WIN_W, WIN_H = 640, 600
        self.window = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock = pygame.time.Clock()

    def array_frame(self):
        surf = pygame.Surface((10 * self.CELL, 20 * self.CELL))
        self._draw_board(surf)
        return pygame.surfarray.array3d(surf)

    def draw(self):
        self.window.fill((15, 17, 26))
        board_surf = pygame.Surface((10 * self.CELL, 20 * self.CELL))
        self._draw_board(board_surf, blit_to=self.window, origin=(180, 40))
        self._draw_piece_box(self.env.hold_piece, (20, 40), "HOLD")
        self._draw_next(origin=(460, 40))
        font = pygame.font.SysFont("consolas", 18)
        self.window.blit(font.render(f"Lines: {self.env.lines}", True, (220,220,220)), (20, 560))
        pygame.display.flip()

    def close(self):
        if self.window:
            pygame.quit()
            self.window = None

    # ── helper drawing routines ─────────────────────────────────
    def _draw_board(self, surf, blit_to=None, origin=(0, 0)):
        CELL = self.CELL
        surf.fill((0, 0, 0))
        for x in range(11):
            pygame.draw.line(surf, (40, 40, 40), (x*CELL, 0), (x*CELL, 20*CELL))
        for y in range(21):
            pygame.draw.line(surf, (40, 40, 40), (0, y*CELL), (10*CELL, y*CELL))
        for row in range(20):
            for col in range(10):
                v = self.env.board[row, col]
                if v:
                    pygame.draw.rect(surf, self.COLORS[v], (col*CELL+1, (19-row)*CELL+1, CELL-2, CELL-2))
        for x, y in self.env._cells(self.env.pos, self.env.rot):
            if 0 <= y < 20:
                pygame.draw.rect(surf, self.COLORS[PIECE_IDS[self.env.current_piece]], (x*CELL+1, (19-y)*CELL+1, CELL-2, CELL-2))
        if blit_to:
            blit_to.blit(surf, origin)

    def _draw_piece_box(self, piece, topleft, label):
        CELL = self.CELL
        pygame.draw.rect(self.window, (255, 255, 255), (*topleft, 5*CELL, 4*CELL), 2)
        font = pygame.font.SysFont("consolas", 18, True)
        self.window.blit(font.render(label, True, (255,255,255)), (topleft[0]+4, topleft[1]-22))
        if piece:
            self._draw_mini(piece, (topleft[0]+CELL, topleft[1]+CELL))

    def _draw_next(self, origin=(460, 40)):
        CELL = self.CELL
        pygame.draw.rect(self.window, (255, 255, 255), (*origin, 5*CELL, 20*CELL), 2)
        font = pygame.font.SysFont("consolas", 18, bold=True)
        self.window.blit(font.render("NEXT", True, (255,255,255)), (origin[0]+4, origin[1]-22))
        for i, p in enumerate(self.env.queue[:5]):
            self._draw_mini(p, (origin[0]+CELL, origin[1]+i*4*CELL))

    def _draw_mini(self, piece, pos):
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
            pygame.draw.rect(self.window, self.COLORS[PIECE_IDS[piece]], (pos[0]+x*CELL, pos[1]+y*CELL, CELL, CELL))
