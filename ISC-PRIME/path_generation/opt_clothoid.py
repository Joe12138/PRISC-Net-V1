import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import numpy as np
import math

from typing import Dict, Union
from scipy import integrate
from scipy.optimize import minimize
from tqdm import tqdm

from util_dir.geometry import get_angle, normalize_angle


def x_func(t):
    return math.cos(t**2)


def y_func(t):
    return math.sin(t**2)


def clothoid_planner(a: float, start_x: float, start_y: float, theta: float, num_p: int = 30):
    x_list = []
    y_list = []

    rot_matrix = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])
    for i in range(num_p):
        theta = (i+1)*2*3*math.pi/360

        # x_output = lambda t: math.cos(t**2)
        x = integrate.quad(x_func, 0, theta)
        y = integrate.quad(y_func, 0, theta)

        # print(x)
        # print(y)
        x_value = x[0]*a
        y_value = y[0]*a

        xy_coord = np.array([[x_value], [y_value]])

        rot_xy = np.dot(rot_matrix, xy_coord)

        x_list.append(start_x+rot_xy[0][0])
        y_list.append(start_y+rot_xy[1][0])

    return x_list, y_list


def obj_function(args: Dict[str, Union[int, float]]):
    target_x = args["target_x"]
    target_y = args["target_y"]
    start_x_copy = args["start_x"]
    start_y_copy  = args["start_y"]
    yaw_copy = args["theta"]

    def obj_value(x, start_x, start_y, yaw, target_x, target_y):
        for j in range(2):
            x_list, y_list = clothoid_planner(x[0], start_x, start_y, yaw, 10)
            start_x = x_list[-1]
            start_y = y_list[-1]
            yaw = normalize_angle(get_angle(np.array([1, 0]), np.array([x_list[-1]-x_list[-2], y_list[-1]-y_list[-2]])))
        x_list, y_list = clothoid_planner(x[0], start_x, start_y, yaw, 10)

        res = (x_list[-1]-target_x)**2+(y_list[-1]-target_y)**2

        # print(x_list[-1], y_list[-1])
        # print("res = {}".format(res))

        return res
    # for j in range(2):
    #     x_list, y_list = clothoid_planner(x[0], start_x, start_y, yaw, 10)
    #     start_x = x_list[-1]
    #     start_y = x_list[-1]

    #     yaw = get_angle(np.array([1, 0]), np.array([x_list[-1]-x_list[-2], y_list[-1]-y_list[-2]]))
    # x_list, y_list = clothoid_planner(x[0], start_x, start_y, yaw, 10)

    # res = (x_list[-1]-target_x)**2+(y_list[-1]-target_y)**2
    return lambda x: obj_value(x, start_x_copy , start_y_copy , yaw_copy, target_x, target_y)


def opt_clothoid_planner(track_xy_full: np.ndarray, yaw: float):
    init_x = np.array([0])

    args = {
        "target_x": track_xy_full[-1][0],
        "target_y": track_xy_full[-1][1],
        "start_x": track_xy_full[9][0],
        "start_y": track_xy_full[9][1],
        "theta": yaw
    }

    cons = (
        {"type": "ineq", "fun": lambda x: 1-x[0]}
    )

    res = minimize(fun=obj_function(args), x0=init_x, method="Nelder-Mead")

    print(res)

    a = res.x[0]
    # a = 5

    start_x = track_xy_full[9][0]
    start_y = track_xy_full[9][1]
    pred_x, pred_y = [], []
    for j in range(3):
        x_list, y_list = clothoid_planner(a, start_x, start_y, yaw, 10)
        start_x = x_list[-1]
        start_y = y_list[-1]
        yaw = normalize_angle(get_angle(np.array([1, 0]), np.array([x_list[-1]-x_list[-2], y_list[-1]-y_list[-2]])))

        pred_x.extend(x_list)
        pred_y.extend(y_list)
    return np.asarray([(x, y) for x, y in zip(pred_x, pred_y)])


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt
    from dataset.pandas_dataset import DatasetPandas, DATA_DICT
    from compute_feature.compute_all_feature_rule import get_target_veh_list
    from util_dir.metric import get_ade

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

            pred_traj = opt_clothoid_planner(track_full_xy, track_full_yaw[9][0])
            ade = get_ade(pred_traj, track_full_xy[10:])
            # print("ade = {}".format(ade))

            plt.plot(track_full_xy[:10, 0], track_full_xy[:10, 1], color="purple", marker="o", zorder=10)
            plt.plot(track_full_xy[10:, 0], track_full_xy[10:, 1], color="orange", marker="o", zorder=10)
            plt.plot(pred_traj[:, 0], pred_traj[:, 1], color="blue", marker="s")

            plt.show()
            plt.cla()
            plt.clf()

            ade_list.append(ade)
            # break
        break

    print("avg-ade = {}, std-ade = {}, min-ade = {}, max-ade = {}".format(np.mean(ade_list), np.std(ade_list),
                                                                          np.min(ade_list), np.max(ade_list)))



