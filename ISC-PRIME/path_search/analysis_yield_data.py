# 分析什么样的初始速度,车辆会跨过Stop line
# 什么样的初始速度, 车辆会停止在Stop line前

import sys
from xmlrpc.client import TRANSPORT_ERROR
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME/")
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from shapely.geometry import LineString, Point

from typing import List
from tqdm import tqdm

from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from hdmap.hd_map import HDMap
from compute_feature.compute_all_feature_rule import get_target_veh_list

from hdmap.visual.map_vis import draw_lanelet_map

def need_to_stop(traffic_sign_list: List[int], hd_map: HDMap):
    for id in traffic_sign_list:
        if "Stop" in hd_map.id_reg_dict[id].sign_meaning:
            return True
    return False

def cross_stop_line(lane_id: int, fut_xy: np.ndarray, hd_map: HDMap):
    for i in range(fut_xy.shape[0]):
        lane_list = hd_map.find_lanelet(pos=fut_xy[i])
        if lane_id in lane_list:
            continue
        else:
            return True
    return False

def is_veh_front(hd_map: HDMap, lane_id: int, data_pandas: DatasetPandas, veh_id_array: np.ndarray, case_id: int, track_id: int, pos_array: np.ndarray):
    current_lanelet = hd_map.id_lanelet_dict[lane_id]

    direct_lane = set(hd_map.graph.following(current_lanelet, withLaneChanges=False))
    direct_lane_id_set = {int(node.id) for node in direct_lane}

    min_dist = -1

    for i in range(veh_id_array.shape[0]):
        veh_id = veh_id_array[i]
        if veh_id == track_id:
            continue

        veh_info_full = data_pandas.get_track_data(case_id=case_id, track_id=veh_id)
        veh_xy_full = veh_info_full[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)

        if veh_xy_full.shape[0] < 10:
            continue
        lane_id_set = set(hd_map.find_lanelet(pos=veh_xy_full[9, :]))
        res = lane_id_set & direct_lane_id_set

        if len(res) > 0:
            dist = np.sqrt((veh_xy_full[9, 0]-pos_array[0])**2+(veh_xy_full[9, 1]-pos_array[1])**2)
            if min_dist < 0:
                min_dist = dist
            elif dist < min_dist:
                min_dist = dist
    if min_dist >= 0:
        return True, min_dist
    else:
        return False, min_dist
        


prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"

mode = "train"
map_path = os.path.join(prefix, "maps")
data_path = os.path.join(prefix, mode)
target_veh_path = os.path.join(prefix, f"{mode}_target_filter_delete_final")

target_file_list = os.listdir(target_veh_path)

plot = False
save_to_file = True

if save_to_file:
    f = open("/home/joe/Desktop/Rule-PRIME/RulePRIME/path_search/train_yield_path_more.csv", "w", encoding="UTF-8")
    f_csv = csv.writer(f)
    head_list = ["scene_name", "case_id", "track_id", "s_vx", "s_vy", "s_speed", "s_yaw", "cross_stop_line", "is_veh_front", "min_dist","dist_stop_line", "e_vx", "e_vy", "e_speed", "e_yaw"]
    f_csv.writerow(head_list)

for file_name in target_file_list:
    scene_name = file_name[:-5]
    if "Intersection" not in scene_name:
        continue
    target_veh_list = get_target_veh_list(target_veh_path=target_veh_path, file_name=file_name)

    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))

    for _, case_id, track_id in tqdm(target_veh_list):
        # DR_USA_Intersection_MA
        # if (scene_name, case_id, track_id) != ("DR_USA_Intersection_GL", 2685, 3):
        #     continue
        case_df = data_pandas.get_case_data(case_id)
        track_id_list = np.unique(case_df[case_df["agent_type"]=="car"].values[:, DATA_DICT["track_id"]])

        # veh_id_array = case_df[case_df["track_id"]]

        # print(track_id_list)
        target_track_info = data_pandas.get_track_data(case_id, track_id)

        target_xy_full = target_track_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
        target_vel_full = target_track_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype(float)
        target_speed = np.hypot(target_vel_full[:, 0], target_vel_full[:, 1]).reshape(-1, 1)
        target_yaw = target_track_info[:, [DATA_DICT["psi_rad"]]].astype(float)

        lanelet_list = hd_map.find_lanelet(pos=target_xy_full[9, :])
        # break_is = False
        for lane_id in lanelet_list:
            traffic_sign_list = hd_map.id_lane_dict[lane_id].traffic_sign
            if need_to_stop(traffic_sign_list, hd_map):
                # print(lane_id)
                lane_obj = hd_map.id_lane_dict[lane_id]
                res = cross_stop_line(lane_id=lane_id, fut_xy=target_xy_full[10:, :], hd_map=hd_map)
                lane_centerline = hd_map.id_lane_dict[lane_id].centerline_array
                cl_ls = LineString(lane_centerline)
                start_point = Point(target_xy_full[9, :])
                dist = cl_ls.project(start_point)

                front_res, min_dist = is_veh_front(hd_map, lane_id, data_pandas, track_id_list, case_id, track_id, target_xy_full[9, :])

                info_list = [scene_name, case_id, track_id, target_vel_full[9, 0], target_vel_full[9, 1], target_speed[9, 0], target_yaw[9, 0],
                res, front_res, min_dist, cl_ls.length-dist, target_vel_full[-1, 0], target_vel_full[-1, 1], target_speed[-1, 0], target_yaw[-1, 0]]

                if save_to_file:
                    f_csv.writerow(info_list)

                if plot:
                    # break_is = True
                    axes = plt.subplot(111)
                    axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)

                    lane_centerline = hd_map.id_lane_dict[lane_id].centerline_array

                    axes.plot(lane_centerline[:, 0], lane_centerline[:, 1], color="#003300")


                    for car_id in track_id_list:
                        if car_id == track_id:
                            continue
                        else:
                            car_track_info = data_pandas.get_track_data(case_id, car_id)
                            car_xy_full = car_track_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
                            if car_xy_full.shape[0] >= 10:
                                axes.plot(car_xy_full[:10, 0], car_xy_full[:10, 1], color="red")
                                axes.scatter(car_xy_full[9, 0], car_xy_full[9, 1], color="red", marker="x", s=15)
                                axes.plot(car_xy_full[10:, 0], car_xy_full[10:, 1], color="green")
                                axes.scatter(car_xy_full[-1, 0], car_xy_full[-1, 1], color="green", marker="x", s=15)
                            else:
                                axes.plot(car_xy_full[:, 0], car_xy_full[:, 1], color="red")
                                axes.scatter(car_xy_full[-1, 0], car_xy_full[-1, 1], color="red", marker="x", s=15)
                                # axes.plot(car_xy_full[10:, 0], car_xy_full[10:, 1], color="green")
                                # axes.scatter(car_xy_full[-1, 0], car_xy_full[-1, 1], color="green", marker="x", s=15)


                    axes.plot(target_xy_full[:10, 0], target_xy_full[:10, 1], color="black")
                    axes.scatter(target_xy_full[9, 0], target_xy_full[9, 1], color="black", marker="o", s=15)
                    axes.plot(target_xy_full[10:, 0], target_xy_full[10:, 1], color="purple")
                    axes.scatter(target_xy_full[-1, 0], target_xy_full[-1, 1], color="purple", marker="o", s=15)
                    # print(target_vel_full[9, :])
                    # print(target_speed[9, :])
                    plt.show()
                    plt.cla()
                    plt.clf()
        #     break
        # if break_is:
        #     break

    # break
if save_to_file:
    f.close()