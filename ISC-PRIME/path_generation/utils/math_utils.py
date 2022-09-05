import math
import numpy as np
from typing import Union


def normalize_angle(angle: Union[float, np.ndarray]):
    angle = angle % (2 * math.pi)
    result = angle - (angle > math.pi) * (2 * math.pi)

    return result


def cal_rot_matrix(yaw: float):
    norm_yaw = normalize_angle(yaw)
    cos_yaw = np.cos(norm_yaw)
    sin_yaw = np.sin(norm_yaw)

    rot_matrix = np.array(
        [[cos_yaw, -sin_yaw],
         [sin_yaw, cos_yaw]]
    )

    return rot_matrix