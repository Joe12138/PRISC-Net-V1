import math

import numpy as np
from scipy.spatial import KDTree

from path_generation.utils.spline_planner import CubicSpline2D
from shapely.geometry import LineString


class ReferenceLine(object):
    def __init__(self, waypoint_x, waypoint_y, wps_step=0.1, obstacles=np.empty((0, 2))):
        self.wps_step = wps_step
        self.obstacles = obstacles

        # Lane Discretization
        self.course_csp = CubicSpline2D(waypoint_x, waypoint_y)
        self.generate_discrete_course()

        coord_list = [(x, y) for x, y in zip(waypoint_x, waypoint_y)]
        self.ref_line_ls = LineString(coordinates=coord_list)
        # print("hello")

    def generate_discrete_course(self):
        self.rs = np.arange(0, self.course_csp.s[-1], self.wps_step)
        self.rx, self.ry = self.course_csp.calc_position(self.rs)
        self.ryaw = self.course_csp.calc_yaw(self.rs)
        self.rkappa = self.course_csp.calc_curvature(self.rs)
        self.rkprime = self.course_csp.calc_curvature_deritative(self.rs)
        self.KDTree = KDTree(np.vstack((self.rx, self.ry)).transpose())

    def get_full_info(self):
        return np.array((self.rs, self.rx, self.ry, self.ryaw, self.rkappa, self.rkprime)).transpose()

    def get_correspond_rpoint(self, global_x, global_y):
        """
        Retrieve info of the waypoint nearest to (global_x, global_y) on the centerline
        :param global_x:
        :param global_y:
        :return:
        """
        dist, id = self.KDTree.query([global_x, global_y])
        # The Corresponding waypoint on the centerline
        rx, ry, ryaw, rkappa, rkappa_prime = self.rx[id], self.ry[id], self.ryaw[id], self.rkappa[id], self.rkprime[id]
        s = self.rs[id]
        d = math.copysign(dist, (global_y-ry)*math.cos(ryaw)-(global_x-rx)*math.sin(ryaw))

        return s, d, rx, ry, ryaw, rkappa, rkappa_prime

    def get_track_sd(self, xy: np.ndarray) -> np.ndarray:
        """
        Given lane, retrieve s&d coordinator of the input xy trajectory.
        :param xy:
        :return:
        """
        sd = np.zeros(xy.shape)

        for i in range(xy.shape[0]):
            dist, id = self.KDTree.query(xy[i], k=1)
            rx, ry, ryaw = self.rx[id], self.ry[id], self.ryaw[id]
            sd[i][0] = self.rs[id]
            sd[i][1] = math.copysign(dist, ((xy[i][1]-ry)*math.cos(ryaw)-(xy[i][0]-rx)*math.sin(ryaw)))
        return sd