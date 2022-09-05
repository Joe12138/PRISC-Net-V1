import math
import logging
import numpy as np
from path_generation.utils.math_utils import normalize_angle


def convert_veh_frenet(yaw, kappa, vel, acc, d, ryaw, rkappa, rkappa_prime):
    delta_yaw = normalize_angle(yaw-ryaw)
    one_minus_rkappa_d = 1 - rkappa * d
    if abs(delta_yaw) >= math.pi/2:
        logging.debug("The delta yaw is large than pi/2")
    if one_minus_rkappa_d <= 0:
        raise RuntimeError("Encounter extreme situation that one_minus_rkappa_d <= 0")
        # logging.warning("Encounter extreme situation that one_minus_rkappa_d <= 0")

    tan_delta_yaw = math.tan(delta_yaw)
    cos_delta_yaw = math.cos(delta_yaw)
    sin_delta_yaw = math.sin(delta_yaw)

    # Derivative with repect to arc length
    d_prime = one_minus_rkappa_d * tan_delta_yaw
    rkappa_d_prime = rkappa_prime * d + rkappa * d_prime
    d_prime_prime = - rkappa_d_prime * tan_delta_yaw + \
                    one_minus_rkappa_d / (cos_delta_yaw) ** 2 * (kappa * one_minus_rkappa_d / cos_delta_yaw - rkappa)
    delta_theta_prime = kappa * one_minus_rkappa_d / cos_delta_yaw - rkappa

    # Derivative with respect to time
    s_d = vel * cos_delta_yaw / one_minus_rkappa_d
    s_dd = (acc * cos_delta_yaw
            - s_d ** 2 * (d_prime * delta_theta_prime - rkappa_d_prime)
            ) / one_minus_rkappa_d
    d_d = vel * sin_delta_yaw
    d_dd = s_dd * d_prime + s_d ** 2 * d_prime_prime

    return s_d, s_dd, d_d, d_dd, d_prime, d_prime_prime


def convert_veh_cartesian(rx, ry, ryaw, rkappa, rkappa_prime, s_d, s_dd, d, d_d, d_dd, d_prime, d_prime_prime,
                          d_derivate_by_prime: bool):
    """
    Given lane info, convert frenet coordinate to cartesian
    :param rx:
    :param ry:
    :param ryaw:
    :param rkappa:
    :param rkappa_prime:
    :param s_d:
    :param s_dd:
    :param d:
    :param d_d:
    :param d_dd:
    :param d_prime:
    :param d_prime_prime:
    :param d_derivate_by_prime:
    :return:
    """
    if d_derivate_by_prime:
        d_dd = None
        d_d = d_prime * s_d
    else:
        d_prime = np.divide(d_d, s_d,
                            out=np.zeros_like(d_d, dtype=float),
                            where=(s_d != 0))
        d_prime_prime = np.divide(d_dd - s_dd * d_prime, s_d ** 2,
                                  out=np.zeros_like(d_dd, dtype=float),
                                  where=(s_d != 0))

    x = rx - d * np.sin(ryaw)
    y = ry + d * np.cos(ryaw)

    # Intermediate variables,
    one_minus_rkappa_d = 1 - rkappa * d
    # Mark: as $ d_prime^2 / one_minus_rkappa_d = tan(delta_yaw)^2 $ is dependent on the symbol of s_d
    #       Therefore when calculating delta_yaw, the angle should +(-) pi under s_d<0 and d_prime<(>)0. Implemented in the 2nd part of the next line.
    delta_yaw = np.arctan2(d_prime, one_minus_rkappa_d) - np.pi * (s_d < 0) * np.sign(d_prime)
    tan_delta_yaw = d_prime / one_minus_rkappa_d
    cos_delta_yaw = np.cos(delta_yaw)

    yaw = normalize_angle(delta_yaw + ryaw)

    rkappa_d_prime = rkappa_prime * d + rkappa * d_prime
    kappa = ((d_prime_prime + rkappa_d_prime * tan_delta_yaw)
             * cos_delta_yaw ** 2 / one_minus_rkappa_d + rkappa) \
            * cos_delta_yaw / one_minus_rkappa_d

    vel = np.sqrt(s_d ** 2 * one_minus_rkappa_d ** 2 + d_d ** 2)

    delta_theta_prime = kappa * one_minus_rkappa_d / cos_delta_yaw - rkappa
    acc = s_dd * one_minus_rkappa_d / cos_delta_yaw + \
          s_d ** 2 / cos_delta_yaw * (d_prime * delta_theta_prime - rkappa_d_prime)

    return x, y, yaw, kappa, vel, acc