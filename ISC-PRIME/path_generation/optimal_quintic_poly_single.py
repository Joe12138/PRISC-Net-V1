import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import math

import numpy as np
from scipy.optimize import minimize

from path_generation.utils.polynomial_planner import QuinticPolynomial


def compute_para(xs, xe, vxs, vxe, axs, axe, ts, te):
    c0 = (
                 2 * (te ** 5) * xs - 2 * (ts ** 5) * xe + 2 * te * (ts ** 5) * vxe - 2 * (te ** 5) * ts * vxs +
                 10 * te * (ts ** 4) * xe - 10 * (te ** 4) * ts * xs - axe * (te ** 2) * (ts ** 5) + 2 * axe * (
                         te ** 3) * (ts ** 4) -
                 axe * (te ** 4) * (ts ** 3) + axs * (te ** 3) * (ts ** 4) - 2 * axs * (te ** 4) * ts ** 3 + axs * (
                         te ** 5) * (ts ** 2) - 10 * (te ** 2) * (ts ** 4) * vxe +
                 8 * (te ** 3) * (ts ** 3) * vxe - 8 * (te ** 3) * (ts ** 3) * vxs + 10 * (te ** 4) * (
                         ts ** 2) * vxs - 20 * (te ** 2) * (ts ** 3) * xe + 20 * (te ** 3) * (ts ** 2) * xs
         ) / (2 * ((te - ts) ** 5))
    c1 = (
                 2 * (te ** 5) * vxs - 2 * ts ** 5 * vxe + 2 * axe * te * ts ** 5 - 2 * axs * te ** 5 * ts +
                 10 * te * ts ** 4 * vxe - 10 * te ** 4 * ts * vxs - axe * te ** 2 * ts ** 4 -
                 4 * axe * te ** 3 * ts ** 3 + 3 * axe * te ** 4 * ts ** 2 - 3 * axs * te ** 2 * ts ** 4 +
                 4 * axs * te ** 3 * ts ** 3 + axs * te ** 4 * ts ** 2 + 16 * te ** 2 * ts ** 3 * vxe -
                 24 * te ** 3 * ts ** 2 * vxe + 24 * te ** 2 * ts ** 3 * vxs - 16 * te ** 3 * ts ** 2 * vxs +
                 60 * te ** 2 * ts ** 2 * xe - 60 * te ** 2 * ts ** 2 * xs
         ) / (2 * ((te - ts) ** 5))
    c2 = (
                 axs * te ** 5 - axe * ts ** 5 - 4 * axe * te * ts ** 4 - 3 * axe * te ** 4 * ts +
                 3 * axs * te * ts ** 4 + 4 * axs * te ** 4 * ts - 36 * te * ts ** 3 * vxe +
                 24 * te ** 3 * ts * vxe - 24 * te * ts ** 3 * vxs + 36 * te ** 3 * ts * vxs -
                 60 * te * ts ** 2 * xe - 60 * te ** 2 * ts * xe + 60 * te * ts ** 2 * xs +
                 60 * te ** 2 * ts * xs + 8 * axe * te ** 2 * ts ** 3 - 8 * axs * te ** 3 * ts ** 2 +
                 12 * te ** 2 * ts ** 2 * vxe - 12 * te ** 2 * ts ** 2 * vxs) / (
                 2 * (te - ts) ** 5)
    c3 = (
                 axe * te ** 4 - 3 * axs * te ** 4 + 3 * axe * ts ** 4 - axs * ts ** 4 - 8 * te ** 3 * vxe -
                 12 * te ** 3 * vxs + 12 * ts ** 3 * vxe + 8 * ts ** 3 * vxs + 20 * te ** 2 * xe -
                 20 * te ** 2 * xs + 20 * ts ** 2 * xe - 20 * ts ** 2 * xs + 4 * axe * te ** 3 * ts -
                 4 * axs * te * ts ** 3 + 28 * te * ts ** 2 * vxe - 32 * te ** 2 * ts * vxe +
                 32 * te * ts ** 2 * vxs - 28 * te ** 2 * ts * vxs - 8 * axe * te ** 2 * ts ** 2 +
                 8 * axs * te ** 2 * ts ** 2 + 80 * te * ts * xe - 80 * te * ts * xs) / (
                 2 * (te - ts) ** 5)
    c4 = -(
            30 * te * xe - 30 * te * xs + 30 * ts * xe - 30 * ts * xs + 2 * axe * te ** 3 - 3 * axs * te ** 3 +
            3 * axe * ts ** 3 - 2 * axs * ts ** 3 - 14 * te ** 2 * vxe - 16 * te ** 2 * vxs + 16 * ts ** 2 * vxe +
            14 * ts ** 2 * vxs - 4 * axe * te * ts ** 2 - axe * te ** 2 * ts + axs * te * ts ** 2 +
            4 * axs * te ** 2 * ts - 2 * te * ts * vxe + 2 * te * ts * vxs) / (
                 2 * (te - ts) ** 5)
    c5 = (
                 12 * xe - 12 * xs - 6 * te * vxe - 6 * te * vxs + 6 * ts * vxe + 6 * ts * vxs +
                 axe * te ** 2 - axs * te ** 2 + axe * ts ** 2 - axs * ts ** 2 - 2 * axe * te * ts +
                 2 * axs * te * ts) / (
                 2 * (te - ts) ** 5)

    return c0, c1, c2, c3, c4, c5


def obj_func(args):
    xs, xe, vxs, axs, ts, te, start_idx, end_idx, his_x_coord = args

    def obj_value(x):
        vxe, axe = x[0], x[1]

        xc0, xc1, xc2, xc3, xc4, xc5 = compute_para(xs, xe, vxs, vxe, axs, axe, ts, te)

        def quintic_planner(idx, c0, c1, c2, c3, c4, c5):
            t = idx * 0.1
            return c0+c1*t+c2*t**2+c3*t**3+c4*t**4+c5*t**5

        result = 0
        for i in range(start_idx, end_idx):
            pred_x = quintic_planner(i, xc0, xc1, xc2, xc3, xc4, xc5)
            result += (pred_x-his_x_coord[i])**2

        return result
    return obj_value


def opt_quintic_polynomial_planner(sx: float, sy: float, syaw: float, sv: float, sa: float,
                                   gx: float, gy: float, xy_obs: np.ndarray, v_obs: np.ndarray,
                                   dt: float = 0.1, T: float = 4):
    init_x_array = np.asarray([0, 0])

    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)

    his_x = xy_obs[:, 0]
    his_y = xy_obs[:, 1]

    x_args = [sx, gx, vxs, axs, 0, T, 0, 10, his_x]
    y_args = [sy, gy, vys, ays, 0, T, 0, 10, his_y]

    cons = (
        {"type": "ineq", "fun": lambda x: x[0] + 15},
        {"type": "ineq", "fun": lambda x: 15 - x[0]},
        {"type": "ineq", "fun": lambda x: x[1] + 5},
        {"type": "ineq", "fun": lambda x: 5 - x[1]}
    )

    x_res = minimize(fun=obj_func(x_args), x0=init_x_array, constraints=cons)
    y_res = minimize(fun=obj_func(y_args), x0=init_x_array, constraints=cons)

    print(x_res.fun, y_res.fun)

    if (not x_res.success) or (not y_res.success):
        raise Exception("No optimal result")

    vxe, axe = x_res.x
    vye, aye = y_res.x

    print(x_res.x)
    print(y_res.x)

    x_quintic_polynomial = QuinticPolynomial(x0=sx, v0=vxs, a0=axs, x1=gx, v1=vxe, a1=axe, time=T-1)
    y_quintic_polynomial = QuinticPolynomial(x0=sy, v0=vys, a0=ays, x1=gy, v1=vye, a1=aye, time=T-1)

    x, y = [], []

    for i in range(40):
        x.append(x_quintic_polynomial.calc_point(0.1 * i))
        y.append(y_quintic_polynomial.calc_point(0.1 * i))
    return x, y


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from dataset.pandas_dataset import DatasetPandas, DATA_DICT
    from compute_feature.compute_all_feature_rule import get_target_veh_list

    mode = "train"
    prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    data_path_prefix = f"{prefix}{mode}"

    target_veh_path = f"{prefix}{mode}_target_filter"

    scene_list = os.listdir(target_veh_path)

    for file_name in scene_list:
        dataset_pandas = DatasetPandas(data_path=os.path.join(data_path_prefix, f"{file_name[:-5]}_{mode}.csv"))
        target_veh_list = get_target_veh_list(target_veh_path=target_veh_path, file_name=file_name)

        for scene_name, case_id, track_id in target_veh_list:
            track_full_info = dataset_pandas.get_track_data(case_id=case_id, track_id=track_id)

            track_full_xy = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
            track_full_v = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype(float)
            track_full_yaw = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

            vxs, vys = track_full_v[0][0], track_full_v[0][0]
            vxn, vyn = track_full_v[1][0], track_full_v[1][0]

            vs = math.sqrt(vxs**2+vys**2)
            vn = math.sqrt(vxn**2+vyn**2)

            a_s = (vn-vs)/0.1

            rx, ry = opt_quintic_polynomial_planner(sx=track_full_xy[0][0],
                                                   sy=track_full_xy[0][1],
                                                   syaw=track_full_yaw[0][0],
                                                   sv=vs,
                                                   sa=a_s,
                                                   gx=track_full_xy[-1][0],
                                                   gy=track_full_xy[-1][1],
                                                   xy_obs=track_full_xy[:10, :],
                                                   dt=0.1,
                                                   T=4)

            x = [i for i in range(40)]
            plt.plot(x, track_full_xy[:, 0], color="purple", marker="o")
            plt.plot(x, rx, color="blue", marker="s")

            plt.show()
        #     break
        # break
