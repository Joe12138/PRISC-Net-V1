"""This module defines all the config parameters."""

FEATURE_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
    "MIN_DISTANCE_FRONT": 6,
    "MIN_DISTANCE_BACK": 7,
    "NUM_NEIGHBORS": 8,
    "OFFSET_FROM_CENTERLINE": 9,
    "DISTANCE_ALONG_CENTERLINE": 10,
}

RAW_DATA_FORMAT = {
    "TIMESTAMP": 0,
    "TRACK_ID": 1,
    "OBJECT_TYPE": 2,
    "X": 3,
    "Y": 4,
    "CITY_NAME": 5,
}

LSTM_HELPER_DICT_IDX = {
    "CENTROIDS": 0,
    "CITY_NAMES": 1,
    "CANDIDATE_CENTERLINES": 2,
    "CANDIDATE_NT_DISTANCES": 3,
    "TRANSLATION": 4,
    "ROTATION": 5,
    "CANDIDATE_DELTA_REFERENCES": 6,
    "DELTA_REFERENCE": 7,
    "SEQ_PATHS": 8,
}

BASELINE_INPUT_FEATURES = {
    "social":
    ["X", "Y", "MIN_DISTANCE_FRONT", "MIN_DISTANCE_BACK", "NUM_NEIGHBORS"],
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "map_social": [
        "OFFSET_FROM_CENTERLINE",
        "DISTANCE_ALONG_CENTERLINE",
        "MIN_DISTANCE_FRONT",
        "MIN_DISTANCE_BACK",
        "NUM_NEIGHBORS",
    ],
    "none": ["X", "Y"],
}

BASELINE_OUTPUT_FEATURES = {
    "social": ["X", "Y"],
    "map": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "map_social": ["OFFSET_FROM_CENTERLINE", "DISTANCE_ALONG_CENTERLINE"],
    "none": ["X", "Y"],
}

# Feature computation
_FEATURES_SMALL_SIZE = 100

# Map Feature computations
_MANHATTAN_INIT_CIRCLE = 15.0           # meters (prev: 10.0)
_MANHATTAN_INCREMENT = 5.0              # meters
_MAX_SEARCH_RADIUS_CENTERLINES = 50.0   # meters
_DFS_THRESHOLD_FRONT_SCALE = 50.0       # (prev: 45.0) 40
_DFS_THRESHOLD_BACK_SCALE = 50.0        # (prev: 40.0) 30
_MIN_DISTANCE_TO_LANEEND = 10.0         # meters

# Centerline Searching threshold
_MAX_TANGENTIAL_BACKWARD_THRESHOLD = 5.0    # meters
_MAX_NORMAL_DISTANCE_TO_CENTERLINE = 15.0   # meters (prev: 15.0)
_NORMAL_DISTANCE_INCREMENT_BOUND = 9.0     # meters (prev:  4.0)
_NORMAL_DISTANCE_INCREMENT = 1.5            # meters (prev:  1.0)

_ENDPT_SEARCH_RADIUS = 40.0                 # meters (preset the search area)
_ENDPT_SEARCH_UPPER_VEL = 16.67             # meters/s (default: 16.67)
_ENDPT_SEARCH_LOWER_VEL = 10.00             # meters/s (default: 10.0)
_ENDPT_SEARCH_SCALE = 2.0                   # meters/s^2
_ENDPT_DIST_THRESHOLD = 1.0                 # meters (prev: 0.5)

_RESAMPLE_FORWARD_DIST = 140.0              # Meter (default: 100.0)
_RESAMPLE_BACKWARD_DIST = 20.0              # Meter (default: 40.0)
_RESAMPLE_INTERVAL = 2.0
_RESAMPLE_BACKWARD = False                  # Whether do centerline extension

_ORACLE_SEARCH_STEP_COUNTS = 10
_ORACLE_SEARCH_NORMAL_BOUNT = 6             # meters
_ORACLE_OFFSET_ERROR = 0.5                  # meters
_MAX_CENTERLINE_CANDIDATES_TEST = 10

_BRIDGING_SMOOTHING_INTERSECTION = 3        # Three points, 6 meters
_BRIDGING_SMOOTHING_T_JUNCTION = 2          # Two points, 4 meters


# Centerline Preprocessing before feeding to planner
_LANE_GAUSSIAN_FILTER = True
_LANE_GAUSSIAN_SIGMA = 1.0


# Social Feature computation (###ATTENTION! when varying observation length)
_FILTER_NBR_STATIONARY = False      # whether to filter those neighbours with velocity lower than STATIONARY_THRESHOLD
_FILTER_NBR_NEAREST_DIST = 100.0    # The maximal nearest distance (meters) with the agent throughout the observation process

_PADDING_COLUMN_FLAG = True # Add Padding column( 0: Origin data 1: Padding data) flag in the neighbor social features
PADDING_TYPE = "REPEAT"     # Padding type for partial sequences
VELOCITY_THRESHOLD = 1.0    # Velocity threshold for stationary (stationary nbrs would not be counted in the social features)
EXIST_THRESHOLD = 10        # Number of timesteps the track should exist to be considered in social context (prev: 15)

STATIONARY_THRESHOLD = 13   # index of the sorted velocity to look at, to call it as stationary (now it's DEPRECATED as the stationary is checked from the maximal vel in the observation period)
DEFAULT_MIN_DIST_FRONT_AND_BACK = 100.0  # Default front/back distance
NEARBY_DISTANCE_THRESHOLD = 50.0  # Distance threshold to call a track as neighbor
FRONT_OR_BACK_OFFSET_THRESHOLD = 5.0  # Offset threshold from direction of travel
