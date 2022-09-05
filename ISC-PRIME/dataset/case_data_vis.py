import sys

import numpy as np

sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import os
import json

import pandas as pd
import matplotlib.pyplot as plt

from dataset.pandas_dataset import DatasetPandas
from hdmap.visual.map_vis import draw_lanelet_map
from hdmap.hd_map import HDMap

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
    print("scene_name = {}".format(scene_name))

    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))
    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    
    case_id_array = np.unique(data_pandas.data_df["case_id"].values)
    print(case_id_array)

    for i in range(case_id_array.shape[0]):
        case_id = int(case_id_array[i])

        print(case_id)
        case_df = data_pandas.get_case_data(case_id)
        track_id_array = np.unique(case_df["track_id"].values)

        axes = plt.subplot(111)
        axes = draw_lanelet_map(hd_map.lanelet_map, axes=axes)

        for j in range(track_id_array.shape[0]):
            track_id = int(track_id_array[j])
            track_full_info = data_pandas.get_track_data(case_id=case_id, track_id=track_id)

            track_xy_full = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]]

            axes.plot(track_xy_full[:, 0], track_xy_full[:, 1], color="purple")
            axes.scatter(track_xy_full[-1, 0], track_xy_full[-1, 1], color="purple", s=15, marker="o")

        print(track_id_array)
        plt.savefig(os.path.join("/home/joe/Desktop/2022ICAL_photo", f"{scene_name}_{case_id}.svg"))
        plt.cla()
        plt.clf()
        # break
        if i > 10:
            break
    # break