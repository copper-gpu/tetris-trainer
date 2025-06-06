from tetris_env import TetrisEnv


def test_reset():
    env = TetrisEnv()
    obs, _ = env.reset()
    assert "board" in obs
