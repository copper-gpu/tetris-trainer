from tetris_env import TetrisEnv
from tetris_env.constants import Action


def test_reset():
    env = TetrisEnv()
    obs, _ = env.reset()
    assert "board" in obs


def test_move_actions():
    env = TetrisEnv()
    env.reset()
    x_before = int(env.pos[0])
    env.step(Action.LEFT)
    assert int(env.pos[0]) == x_before - 1
    env.step(Action.RIGHT)
    assert int(env.pos[0]) == x_before


def test_hard_drop_locks_piece():
    env = TetrisEnv()
    env.reset()
    _, reward, done, _, _ = env.step(Action.HARD_DROP)
    assert reward > 0
    assert not done
    assert int(env.pos[1]) == env.HEIGHT + 1


def test_game_over_detection():
    env = TetrisEnv()
    env.reset()
    env.board[-env.BUFFER:, :] = 0
    env.board[-1, 0] = 1
    _, _, done, _, _ = env.step(Action.NONE)
    assert done
