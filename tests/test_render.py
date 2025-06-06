import os
import sys
import numpy as np

# Ensure the repository root is on the import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tetris_env import TetrisEnv

def test_render_rgb_array_shape():
    env = TetrisEnv()
    env.reset()
    frame = env.render("rgb_array")
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (240, 480, 3)
    assert frame.dtype == np.uint8
    env.close()
