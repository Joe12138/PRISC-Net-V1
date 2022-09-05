import sys

import numpy as np

sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point
from dataset.pandas_dataset import DatasetPandas
from hdmap.visual.map_vis import draw_lanelet_map
from hdmap.hd_map import HDMap
from path_search.search_with_rule_v2 import path_search_rule
from path_search.visual.path_viz import plot_path, plot_path_area

DATA_DICT = {
    "case_id": 0,
    "track_id": 1,
    "frame_id": 2,
    "timestamp_ms": 3,
    "agent_type": 4,
    "x": 5,
    "y": 6,
    "vx": 7,
    "vy": 8,
    "psi_rad": 9,
    "length": 10,
    "width": 11,
    "track_to_predict": 12
}

mode = "train"
path_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2"
map_path = os.path.join(path_prefix, "maps")
data_path = os.path.join(path_prefix, mode)
target_veh_path = os.path.join(path_prefix, f"{mode}_target_filter")

target_list = os.listdir(target_veh_path)

for file_name in target_list:
    scene_name = file_name[:-5]
    # if scene_name != "DR_USA_Roundabout_FT":
    #     continue
    print("scene_name = {}".format(scene_name))

    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))
    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    
    case_id_array = np.unique(data_pandas.data_df["case_id"].values)
    # print(case_id_array)

    for i in range(case_id_array.shape[0]):
        case_id = int(case_id_array[i])
        if case_id >= 3:
            break

        print(case_id)
        case_df = data_pandas.get_case_data(case_id)
        track_id_array = np.unique(case_df["track_id"].values)

        for j in range(0, track_id_array.shape[0]):
            
            min_x = 10e9
            min_y = 10e9
            max_x = -10e9
            max_y = -10e9
            track_id = int(track_id_array[j])
            track_full_info = data_pandas.get_track_data(case_id=case_id, track_id=track_id)

            if track_full_info.shape[0] != 40:
                continue

            track_xy_full = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]]
            track_yaw_full = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

            cl_list, path_list = path_search_rule(
                track_obs_xy=track_xy_full[:10],
                track_obs_yaw=track_yaw_full[:10],
                case_data=case_df,
                track_id=track_id,
                hd_map=hd_map,
                roundabout=True if "Roundabout" in scene_name else False
            )

            axes = plt.subplot(111)
            axes = draw_lanelet_map(hd_map.lanelet_map, axes=axes)
            for idx, path in enumerate(path_list):
                # axes = plot_path(path=path, id_lane_dict=hd_map.id_lane_dict, axes=axes, color="gray")
                axes = plot_path_area(cl=cl_list[idx], start_point=track_xy_full[9], path=path, id_lane_dict=hd_map.id_lane_dict, hd_map=hd_map, axes=axes)
                axes.plot(cl_list[idx][:, 0], cl_list[idx][:, 1], color="#008B8B")
            axes.plot(track_xy_full[:10, 0], track_xy_full[:10, 1], color="#006400")
            axes.scatter(track_xy_full[9, 0], track_xy_full[9, 1], color="#006400", marker="x", s=15)
            
            axes.plot(track_xy_full[10:, 0], track_xy_full[10:, 1], color="#DC143C")
            axes.scatter(track_xy_full[-1, 0], track_xy_full[-1, 1], color="#DC143C", marker="x", s=15)
            axes.set_aspect(1)
            # plt.show()
            plt.savefig("/home/joe/Desktop/2022ICAL_photo/path_search/{}_{}_{}.svg".format(scene_name, case_id, track_id))
            plt.cla()
            plt.clf()
    #         break
    #     break
    break
