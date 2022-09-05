class FrenetTrajectory(object):
    def __init__(self):
        self.t = None

        # d--lateral offset relative to centerline
        self.d = None
        self.d_d = None
        self.d_dd = None
        # or dependent with the arc length
        self.d_prime = None
        self.d_prime_prime = None

        # s--arc length along centerline
        self.s = None
        self.s_d = None
        self.s_dd = None

        # vehicle's global information
        self.x = None           # global x
        self.y = None           # global y
        self.yaw = None         # heading angle
        self.kappa = None       # curvature
        self.vel = None         # velocity
        self.acc = None         # acceleration
        self.dis = None         # distance between points
        self.exceed_lane_range = False

        # cost parameters for optimal planning (lateral offset, target velocity, and total lost)
        self.c_d = 0.0
        self.c_vel = 0.0
        self.c_jerk = 0.0
        self.c_total = 0.0