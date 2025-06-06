from __future__ import annotations

import gymnasium as gym

from .core import TetrisCore
from .renderer import PygameRenderer
from .constants import Action

class TetrisEnv(TetrisCore, gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, seed: int | None = None):
        TetrisCore.__init__(self, seed)
        self.renderer: PygameRenderer | None = None
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Dict({
            "board":   gym.spaces.Box(0, 7, (20, 10), float),
            "current": gym.spaces.Discrete(8),
            "pos":     gym.spaces.Box(0, 20, (2,), int),
            "rot":     gym.spaces.Discrete(4),
            "hold":    gym.spaces.Discrete(8),
            "next":    gym.spaces.Box(0, 7, (5,), int),
            "lines":   gym.spaces.Discrete(10000),
        })
        self.render_mode = None

    # ── Reset ─────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self.random.seed(seed)
        obs, info = super().reset()
        if self.render_mode == "human":
            if self.renderer is None:
                self.renderer = PygameRenderer(self)
                self.renderer.setup()
            self.renderer.draw()
        return obs, info

    # ── Step ──────────────────────────────────────────────────────
    def step(self, action):
        obs, reward, done, _, info = super().step(action)
        if self.render_mode == "human" and self.renderer is not None:
            self.renderer.draw()
            self.renderer.clock.tick(self.metadata["render_fps"])
        return obs, reward, done, False, info

    # ── Render ─────────────────────────────────────────────────--
    def render(self, mode="human"):
        self.render_mode = mode
        if mode == "human":
            if self.renderer is None:
                self.renderer = PygameRenderer(self)
                self.renderer.setup()
            self.renderer.draw()
        elif mode == "rgb_array":
            if self.renderer is None:
                self.renderer = PygameRenderer(self)
                self.renderer.setup()
            return self.renderer.array_frame()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
