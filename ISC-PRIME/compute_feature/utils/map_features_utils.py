from typing import Union, List
from shapely.geometry import Point, LineString

import logging
import numpy as np


from compute_feature.utils.baseline_config import _ORACLE_SEARCH_STEP_COUNTS, _ORACLE_SEARCH_NORMAL_BOUNT, _ORACLE_OFFSET_ERROR


def get_oracle_centerline(xy_fut: Union[np.ndarray, None],
                          centerlines: List[np.ndarray],
                          last_n_stamps: int = _ORACLE_SEARCH_STEP_COUNTS):
    '''
    Get the oracle centerline id from calculated nt distance of the GT future traj.
    Args:
        xy_fut: the trajectory after the observed length
        centerlines: the given candidates.
        last_n_stamps: the number of steps to be counted.
    Return:
        the oracle centerline id
        probablity
    '''

    oracle_centerline_idx = []
    if xy_fut is None: # Under Test mode
        return None, None

    xy = xy_fut[-last_n_stamps:]
    # 1. First use distance along centerline with threshold 80% , can not filter parallel lane
    max_dist_along_cl = -float("inf")
    oracle_centerlines = []
    oracle_indices = []
    probability_of_cls = [0] * len(centerlines)
    for cl_idx, centerline in enumerate(centerlines):
        cl_ls = LineString(centerline)
        start_dist = cl_ls.project(Point(xy[0, 0], xy[0, 1]))
        end_dist = cl_ls.project(Point(xy[-1, 0], xy[-1, 1]))
        dist_along_cl = end_dist - start_dist
        # Set tolerance error 20 percents
        if dist_along_cl > max_dist_along_cl * 0.8:
            max_dist_along_cl = max(dist_along_cl, max_dist_along_cl)
            oracle_centerlines.append(centerline)
            oracle_indices.append(cl_idx)

    # 2. Score based on offset with acceptable offset error _ORACLE_OFFSET_ERROR=0.5
    offset_list = []
    for idx, centerline in enumerate(oracle_centerlines):
        max_offset = 0.0
        for loc in xy:
            offset = Point(loc).distance(LineString(centerline))
            max_offset = max(offset, max_offset)
        offset_list.append(max_offset)
    min_of_max_offset = min(offset_list)
    if min_of_max_offset > _ORACLE_SEARCH_NORMAL_BOUNT:
        logging.warning("The max offset of the best oracle is {:.2f}".format(min_of_max_offset))
    probablity_of_oracle = [1 if (offset-min(offset_list)) <= _ORACLE_OFFSET_ERROR else 0 for offset in offset_list]
    for idx, prob in enumerate(probablity_of_oracle):
        if prob == 1:
            probability_of_cls[oracle_indices[idx]] = 1
            oracle_centerline_idx.append(oracle_indices[idx])

    # Normalize the probablities
    probability_of_cls[:] = [prob / sum(probability_of_cls) for prob in probability_of_cls]

    return oracle_centerline_idx, probability_of_cls