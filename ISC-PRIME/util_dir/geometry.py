import math
import numpy as np


def get_angle(vec_a: np.ndarray, vec_b: np.ndarray):
    """
    Compute the angle from a to b.
    :param vec_a:
    :param vec_b:
    :return:
    """
    res = -math.atan2(vec_a[1], vec_a[0])+math.atan2(vec_b[1], vec_b[0])
    if res > math.pi:
        res -= 2 * math.pi
    elif res < -math.pi:
        res += 2 * math.pi

    return res


def normalize_angle(angle: float):
    a = math.fmod(angle+math.pi, 2*math.pi)

    if a < 0:
        a += 2 * math.pi

    return a-math.pi


def is_on_line(target_p: np.ndarray, start_p: np.ndarray, end_p: np.ndarray) -> bool:
    """
    If a point is on the line which starts with start_p and end with end_p.
    :param target_p: The coordinate of target point.
    :param start_p: The coordinate of start point.
    :param end_p: The coordinate of end point.
    :return: bool value
    """
    q_x, q_y = target_p[0], target_p[1]
    a_x, a_y = start_p[0], start_p[1]
    b_x, b_y = end_p[0], end_p[1]
    line_len = np.linalg.norm(end_p-start_p)

    t = ((a_x - b_x) * (a_x - q_x) + (a_y - b_y) * (a_y - q_y)) / (line_len * line_len)
    if 0 <= t <= 1:
        return True
    else:
        return False
