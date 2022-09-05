import math


class VehicleState(object):
    def __init__(self, x: float, y: float, vx: float, vy: float, acc: float,
                 yaw: float, kappa: float, width: float, length: float):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.yaw = yaw
        self.kappa = kappa
        self.width = width
        self.length = length

        self.vel = math.hypot(self.vx, self.vy)
        self.acc = acc
        # coordinate in feature frame would be initialized when the lane is given
        self.s = None
        self.s_d = None
        self.s_dd = None
        self.d = None
        self.d_d = None
        self.d_dd = None

        # d could be sampled dependent on time (_d) or arc length (_prime)
        self.d_prime = None
        self.d_prime_prime = None

    def get_global_info(self):
        return self.x, self.y, self.yaw, self.kappa, self.vel, self.acc

    def set_frenet_info(self, s, s_d, s_dd, d, d_d, d_dd, d_prime=None, d_prime_prime=None):
        self.s = s
        self.s_d = s_d
        self.s_dd = s_dd
        self.d = d
        self.d_d = d_d
        self.d_dd = d_dd
        self.d_prime = d_prime
        self.d_prime_prime = d_prime_prime