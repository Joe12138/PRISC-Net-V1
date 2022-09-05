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
    if scene_name != "DR_USA_Roundabout_FT":
        continue
    print("scene_name = {}".format(scene_name))

    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))
    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    
    case_id_array = np.unique(data_pandas.data_df["case_id"].values)
    print(case_id_array)

    for i in range(case_id_array.shape[0]):
        

        case_id = int(case_id_array[i])
        if case_id != 7:
            continue

        print(case_id)
        case_df = data_pandas.get_case_data(case_id)
        track_id_array = np.unique(case_df["track_id"].values)

        axes = plt.subplot(111)
        # axes = draw_lanelet_map(hd_map.lanelet_map, axes=axes)
        # for lane_id, lane_obj in hd_map.id_lane_dict.items():
        #     center_array = lane_obj.centerline_array
        #     cl_ls = LineString(center_array)

        #     equal_array = [(center_array[0][0], center_array[0][1])]
        #     length = 5
        #     while length < cl_ls.length:
        #         point = cl_ls.interpolate(length)
        #         equal_array.append((point.x, point.y))
        #         length += 5
            
        #     equal_array = np.asarray(equal_array)

        #     if np.min(equal_array[:, 0]) < min_x:
        #         min_x = np.min(equal_array[:, 0])
        #     if np.max(equal_array[:, 0]) > max_x:
        #         max_x = np.max(equal_array[:, 0])
            
        #     if np.min(equal_array[:, 1]) < min_y:
        #         min_y = np.min(equal_array[:, 1])
        #     if np.max(equal_array[:, 1]) > max_y:
        #         max_y = np.max(equal_array[:, 1])

        #     for k in range(equal_array.shape[0]-1):
        #         axes.annotate("", xy=equal_array[k+1], xytext=equal_array[k], arrowprops=dict(arrowstyle="->", color="black"))

        for j in range(7, track_id_array.shape[0]):
            min_x = 10e9
            min_y = 10e9
            max_x = -10e9
            max_y = -10e9
            track_id = int(track_id_array[j])
            track_full_info = data_pandas.get_track_data(case_id=case_id, track_id=track_id)

            track_xy_full = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]]
            track_yaw_full = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

            # draw path search result
            
            if track_xy_full.shape[0] == 40:
                # traj_ls = LineString(track_xy_full)
                # if traj_ls.length < 5:
                #     continue
                cl_list, _ = path_search_rule(
                    track_obs_xy=track_xy_full[:10],
                    track_obs_yaw=track_yaw_full[:10],
                    case_data=case_df,
                    track_id=track_id,
                    hd_map=hd_map,
                    roundabout=True if "Roundabout" in scene_name else False
                )

                for idx, cl in enumerate(cl_list):
                    # if idx > 1: 
                    #     break

                    if np.min(cl[:, 0]) < min_x:
                        min_x = np.min(cl[:, 0])
                    if np.max(cl[:, 0]) > max_x:
                        max_x = np.max(cl[:, 0])
                    
                    if np.min(cl[:, 1]) < min_y:
                        min_y = np.min(cl[:, 1])
                    if np.max(cl[:, 1]) > max_y:
                        max_y = np.max(cl[:, 1])
                    axes.plot(cl[:, 0], cl[:, 1], color="black")
                    cl_ls = LineString(cl)

                    # dist = 2
                    # while dist < cl_ls.length:
                    #     point = cl_ls.interpolate(dist)
                    #     axes.scatter(point.x, point.y, color="purple", marker="o", s=15)
                    #     dist += 2
                
                axes.plot(track_xy_full[:10, 0], track_xy_full[:10, 1], color="#006400")
                axes.scatter(track_xy_full[9, 0], track_xy_full[9, 1], color="#006400", marker="x", s=15)
                
                axes.plot(track_xy_full[10:, 0], track_xy_full[10:, 1], color="#DC143C")
                axes.scatter(track_xy_full[-1, 0], track_xy_full[-1, 1], color="#DC143C", marker="x", s=15)
                axes.set_aspect(1)
                plt.xlim((min_x, max_x))
                plt.ylim((min_y, max_y))
                plt.savefig(os.path.join("/home/joe/Desktop/2022ICAL_photo", f"{scene_name}_{case_id}_path_search_no_sample.svg"))
                
                # plt.show()
                plt.cla()
                plt.clf()

                exit(0)
            


            # for p in range(track_xy_full.shape[0]-1):
            #     print(track_xy_full[p+1][:])
            #     axes.annotate("", xy=track_xy_full[p+1][:], xytext=track_xy_full[p][:], arrowprops=dict(arrowstyle="->", color="purple"))

            # if np.min(track_xy_full[:, 0]) < min_x:
            #     min_x = np.min(track_xy_full[:, 0])
            # if np.max(track_xy_full[:, 0]) > max_x:
            #     max_x = np.max(track_xy_full[:, 0])
            
            # if np.min(track_xy_full[:, 1]) < min_y:
            #     min_y = np.min(track_xy_full[:, 1])
            # if np.max(track_xy_full[:, 1]) > max_y:
            #     max_y = np.max(track_xy_full[:, 1])

            # axes.plot(track_xy_full[:, 0], track_xy_full[:, 1], color="purple")
            # axes.scatter(track_xy_full[-1, 0], track_xy_full[-1, 1], color="purple", s=15, marker="o")

        # print(track_id_array)
        # axes.set_aspect(1)
        # # plt.xlim((min_x, max_x))
        # # plt.ylim((min_y, max_y))
        # # plt.savefig(os.path.join("/home/joe/Desktop/2022ICAL_photo", f"{scene_name}_{case_id}_vec.svg"))
        
        # plt.show()
        # plt.cla()
        # plt.clf()
        # # break
        # if i > 10:
        #     break
    # break