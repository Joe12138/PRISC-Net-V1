##############################  Lane  ############################################
_LANE_WIDTH = 5.0
_WAYPOINTS_STEP = 0.1


##############################  Vehicle  ############################################
#TODO: Add max_steer angle constrain which is related to vehicle's wheelbase
#TODO: Physical parameters corresponding to different types of vehicles

_VEHICLE_TYPE = {
    # id: [speed,   acc,    dec,    curvature,  collision_r]
    1: [108/3.6,   6.0,    -6.0,    1.0/3.0,    2.0],
    2: [50.0/3.6,   4.0,    -6.0,    1.0,        2.0],
}

# State estimation
_VEL_REGULIZER = 90 / 3.6
_ACC_REGULIZER = 3.0
_DEC_REGULIZER = -3.0

