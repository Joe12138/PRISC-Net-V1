"""This module defines all the config parameters."""

## Datasets -----------------------------------------------------------------
FEATURE_FILE_PREFIX = "seq_"
_CL_SEGMENT_NUM = 81

## Sampling Strategy (Activate by uncomment the item)
_SCALING_NUM = 80
_SCALING_S = 160
_SCALING_D = 5
_SCALING_RANGE = [0.75, 1.25]
_DELTA_YAW_RANGE = _OBS_VARIANCE = _OBS_DROPOUT = None

## Fixed parameters --------------------------------------------------------
_MAX_GUESSES_NUM = 6
_MISSING_THRESHOLD = 2.0

## Loss Function -----------------------------------------------------------
_RANK_LOSS_WEIGHT = 10.0  # used when the best ranking is counted
_LANE_LOSS_WEIGHT = 10.0  # used in lane-traj-level loss

# Vis ----------------------------------------------------------------------
_MAX_VIS_GUESSES_NUM = 6
_HISTOGRAM_BINS = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4, 5, 10, 15, 20, 1000]