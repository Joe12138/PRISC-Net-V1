import numpy as np

from path_generation.object.vehicle_state import VehicleState
from path_generation.utils.frenet_utils import FrenetTrajectory
from path_generation.utils.polynomial_planner import QuarticPolynomial, QuinticPolynomial
from path_generation.object.reference_line import ReferenceLine
from path_generation.utils.generator_utils import convert_veh_frenet, convert_veh_cartesian


class FrenetPlanner(object):
    def __init__(self, veh_state: VehicleState, plan_time: int, plan_dt: float):
        self.plan_time = plan_time
        self.plan_dt = plan_dt

        self.veh_state = veh_state

        self.refer_line = None

        self.kd = 1.0
        self.kv = 1.0
        self.kj = 0.1

    def init_by_target_lane(self, refer_line: ReferenceLine):
        self.refer_line = refer_line
        x, y, yaw, kappa, vel, acc = self.veh_state.get_global_info()

        s, d, rx, ry, ryaw, rkappa, rkappa_prime = self.refer_line.get_correspond_rpoint(x, y)
        s_d, s_dd, d_d, d_dd, d_prime, d_prime_prime = convert_veh_frenet(yaw, kappa, vel, acc, d, ryaw, rkappa, rkappa_prime)

        self.veh_state.set_frenet_info(s, s_d, s_dd, d, d_d, d_dd, d_prime, d_prime_prime)

    def frenet_planning(self):
        frenet_trajs = []

    def set_traj_global_info(self, traj: FrenetTrajectory):
        """
        Set the global information for the given trajectory
        :param traj:
        :return:
        """
        rx, ry, ryaw, rkappa, rkappa_prime = self.refer_line.course_csp.calc_all_in_single_forward(traj.s)
        deriv_by_prime = (traj.d_prime is not None) and (traj.d_prime_prime is not None)
        traj.x, traj.y, traj.yaw, traj.kappa, traj.vel, traj.acc = convert_veh_cartesian(
            rx, ry, ryaw, rkappa, rkappa_prime, traj.s_d, traj.s_dd, traj.d, traj.d_d, traj.d_dd, traj.d_prime,
            traj.d_prime_prime, d_derivate_by_prime=deriv_by_prime
        )
        traj.dis = np.hypot(np.diff(np.array(traj.x)), np.diff(np.array(traj.y)))

    def trajectory_generation(self, lon_vel: float, lat_d: float, lon_acc: float = 0.0, lat_vec: float = 0.0,
                              lat_acc: float = 0.0):
        traj = FrenetTrajectory()

        lon_qp = QuarticPolynomial(x0=self.veh_state.s,
                                   v0=self.veh_state.s_d,
                                   a0=self.veh_state.s_dd,
                                   v1=lon_vel,
                                   a1=lon_acc,
                                   time=self.plan_time)

        traj.t = np.arange(0.0, self.plan_time, self.plan_dt) + self.plan_dt
        traj.s = lon_qp.calc_point(traj.t)
        traj.s_d = lon_qp.calc_first_derivative(traj.t)
        traj.s_dd = lon_qp.calc_second_derivative(traj.t)
        traj_s_ddd = lon_qp.calc_third_derivative(traj.t)

        lat_qp = QuinticPolynomial(x0=self.veh_state.d,
                                   v0=self.veh_state.d_d,
                                   a0=self.veh_state.d_dd,
                                   x1=lat_d,
                                   v1=lat_vec,
                                   a1=lat_acc,
                                   time=self.plan_time)

        traj.d = lat_qp.calc_point(traj.t)
        traj.d_d = lat_qp.calc_first_derivative(traj.t)
        traj.d_dd = lat_qp.calc_second_derivative(traj.t)
        traj_d_ddd = lat_qp.calc_third_derivative(traj.t)

        self.set_traj_global_info(traj)

        traj.c_d = self.kd * traj.d[-1] ** 2
        traj.c_vel = (traj.vel[-1] - self.veh_state.vel) ** 2
        traj.c_jerk = sum(np.power(traj_d_ddd, 2)) + sum(np.power(traj_s_ddd, 2))
        traj.c_total = self.kd*traj.c_d + self.kv*traj.c_vel + self.kj*traj.c_jerk

        return traj

