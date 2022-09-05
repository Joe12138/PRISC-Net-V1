import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import math

import numpy as np
from scipy.optimize import minimize
from typing import Dict, Union

from path_generation.quintic_generation import quintic_polynomial_planner
# from compute_feature.compute_all_feature_rule import get_init_veh_state

from util_dir.geometry import normalize_angle, get_angle
from util_dir.metric import get_ade, get_fde

from path_generation.object.reference_line import ReferenceLine

from tqdm import tqdm


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


def quintic_planner(idx, c0, c1, c2, c3, c4, c5):
    t = idx * 0.1
    return c0 + c1 * t + c2 * t ** 2 + c3 * t ** 3 + c4 * t ** 4 + c5 * t ** 5


def yaw_change(args: Dict[str, Union[int, float]]):
    sx, sy, ex, ey = args["sx"], args["sy"], args["ex"], args["ey"]
    svx, svy = args["svx"], args["svy"]
    sax, say = args["sax"], args["say"]
    ts, te = args["ts"], args["te"]
    num_idx = args["num_idx"]

    ref_line = np.array([[sx, sy], [ex, ey]])

    ref_line_obj = ReferenceLine(waypoint_x=ref_line[:, 0], waypoint_y=ref_line[:, 1])

    def obj_value(x):
        ev, ea, eyaw = x[0], x[1], x[2]
        vxe = ev * math.cos(eyaw)
        vye = ev * math.sin(eyaw)

        axe = ea * math.cos(eyaw)
        aye = ea * math.sin(eyaw)

        xc0, xc1, xc2, xc3, xc4, xc5 = compute_para(xs=sx, xe=ex, vxs=svx, vxe=vxe, axs=sax, axe=axe, ts=ts, te=te)
        yc0, yc1, yc2, yc3, yc4, yc5 = compute_para(xs=sy, xe=ey, vxs=svy, vxe=vye, axs=say, axe=aye, ts=ts, te=te)

        x_list = []
        y_list = []

        for i in range(num_idx):
            x_list.append(quintic_planner(i+1, xc0, xc1, xc2, xc3, xc4, xc5))
            y_list.append(quintic_planner(i+1, yc0, yc1, yc2, yc3, yc4, yc5))

        # xy_array = np.asarray([(x, y) for x, y in zip(x_list, y_list)])
        # sd_array = ref_line_obj.get_track_sd(xy_array)
        # for x, y in zip(x_list, y_list):

        result = 0
        for i in range(num_idx):
            if i == 0:
                direct_array = np.array([x_list[i+1]-x_list[i], y_list[i+1]-y_list[i]])
                traj_angle = get_angle(direct_array, np.array([svx, svy]))
                result += abs(traj_angle)
            elif i == num_idx-1:
                direct_array = np.array([x_list[i] - x_list[i-1], y_list[i] - y_list[i-1]])
                last_direct_array = np.array([x_list[i-1] - x_list[i-2], y_list[i-1] - y_list[i-2]])
                traj_angle = get_angle(direct_array, last_direct_array)
                result += abs(traj_angle)
            else:
                direct_array = np.array([x_list[i+1] - x_list[i], y_list[i+1] - y_list[i]])
                last_direct_array = np.array([x_list[i] - x_list[i-1], y_list[i] - y_list[i-1]])
                traj_angle = get_angle(direct_array, last_direct_array)
                result += abs(traj_angle)
        # result += np.sum(np.abs(sd_array[:, 1]))
        return result
    return obj_value

def traj_length(args: Dict[str, Union[int, float]]):
    sx, sy, ex, ey = args["sx"], args["sy"], args["ex"], args["ey"]
    svx, svy = args["svx"], args["svy"]
    sax, say = args["sax"], args["say"]
    ts, te = args["ts"], args["te"]
    num_idx = args["num_idx"]

    def obj_value(x):
        ev, ea, eyaw = x[0], x[1], x[2]
        vxe = ev * math.cos(eyaw)
        vye = ev * math.sin(eyaw)

        axe = ea * math.cos(eyaw)
        aye = ea * math.sin(eyaw)

        xc0, xc1, xc2, xc3, xc4, xc5 = compute_para(xs=sx, xe=ex, vxs=svx, vxe=vxe, axs=sax, axe=axe, ts=ts, te=te)
        yc0, yc1, yc2, yc3, yc4, yc5 = compute_para(xs=sy, xe=ey, vxs=svy, vxe=vye, axs=say, axe=aye, ts=ts, te=te)

        x_list = []
        y_list = []

        for i in range(num_idx):
            x_list.append(quintic_planner(i+1, xc0, xc1, xc2, xc3, xc4, xc5))
            y_list.append(quintic_planner(i+1, yc0, yc1, yc2, yc3, yc4, yc5))

        dist = 0
        for i in range(len(x_list)-1):
            dist += math.sqrt((x_list[i+1]-x_list[i])**2+(y_list[i+1]-y_list[i])**2)
        
        return dist
    
    return obj_value


def opt_quintic_polynomial_planner(sx: float, sy: float, syaw: float, sv: float, sa: float,
                                   gx: float, gy: float, T: float = 3):
    vxs = sv * math.cos(syaw)
    vys = sv * math.sin(syaw)
    axs = sa * math.cos(syaw)
    ays = sa * math.sin(syaw)

    args = {
        "sx": sx, "sy": sy, "ex": gx, "ey": gy,
        "svx": vxs, "svy": vys, "sax": axs, "say": ays,
        "ts": 0, "te": 3, "num_idx": 30
    }

    init_x_array = np.asarray([0, 0, 0])

    res = minimize(fun=traj_length(args), x0=init_x_array, method="SLSQP")

    # print(res.fun)

    print(res.success)

    # # if not res.success:
    # #     raise Exception("No optimal result")
    print(res.x)
    # print(res.message)
    ve, ae, yaw_e = res.x

    rx, ry = quintic_polynomial_planner(sx=sx,
                                        sy=sy,
                                        syaw=syaw,
                                        sv=sv,
                                        sa=sa,
                                        gx=gx,
                                        gy=gy,
                                        gyaw=yaw_e,
                                        gv=ve,
                                        ga=ae,
                                        T=3)

    return rx, ry


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from dataset.pandas_dataset import DatasetPandas, DATA_DICT
    from compute_feature.compute_all_feature_rule import get_target_veh_list

    mode = "val"
    prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    data_path_prefix = f"{prefix}{mode}"

    target_veh_path = f"{prefix}{mode}_target_filter_delete_final"

    scene_list = os.listdir(target_veh_path)

    ade_list = []

    for file_name in scene_list:
        print(file_name)
        dataset_pandas = DatasetPandas(data_path=os.path.join(data_path_prefix, f"{file_name[:-5]}_{mode}.csv"))
        target_veh_list = get_target_veh_list(target_veh_path=target_veh_path, file_name=file_name)

        for scene_name, case_id, track_id in tqdm(target_veh_list):
            track_full_info = dataset_pandas.get_track_data(case_id=case_id, track_id=track_id)

            track_full_xy = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
            track_full_v = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype(float)
            track_full_yaw = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

            vxs, vys = track_full_v[9][0], track_full_v[9][1]
            vxn, vyn = track_full_v[10][0], track_full_v[10][1]

            vs = math.sqrt(vxs ** 2 + vys ** 2)
            vn = math.sqrt(vxn ** 2 + vyn ** 2)

            a_s = (vn - vs) / 0.1
            end_v = math.sqrt(track_full_v[-1][0] ** 2 + track_full_info[-1][1] ** 2)
            l_end_v = math.sqrt(track_full_v[-2][0] ** 2 + track_full_info[-2][1] ** 2)

            rx, ry = opt_quintic_polynomial_planner(sx=track_full_xy[9][0],
                                                    sy=track_full_xy[9][1],
                                                    syaw=track_full_yaw[9][0],
                                                    sv=vs,
                                                    sa=a_s,
                                                    gx=track_full_xy[-1][0],
                                                    gy=track_full_xy[-1][1],
                                                    T=3)
            pred_traj = np.asarray([(x, y) for x, y in zip(rx, ry)])
            ade = get_ade(pred_traj, track_full_xy[10:])
            # print("ade = {}".format(ade))

            plt.plot(track_full_xy[:10, 0], track_full_xy[:10, 1], color="purple", marker="o", zorder=10)
            plt.plot(track_full_xy[10:, 0], track_full_xy[10:, 1], color="orange", marker="o", zorder=10)
            plt.plot(rx, ry, color="blue", marker="s")
            
            plt.show()
            plt.cla()
            plt.clf()
            
            ade_list.append(ade)
            # break
        # break

    print("avg-ade = {}, std-ade = {}, min-ade = {}, max-ade = {}".format(np.mean(ade_list), np.std(ade_list),
                                                                          np.min(ade_list), np.max(ade_list)))