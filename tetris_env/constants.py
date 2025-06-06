from enum import IntEnum

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
BUFFER_ROWS = 4

class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    ROT_CW = 2
    ROT_CCW = 3
    SOFT_DROP = 4
    HARD_DROP = 5
    HOLD = 6
    NONE = 7
