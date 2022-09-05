import sys
import math
import pandas

sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import numpy as np
import json
import os
import pandas as pd
from tqdm import tqdm
from typing import List
from joblib import Parallel, delayed
from shapely.geometry import LineString, Point

from hdmap.hd_map import HDMap
from dataset.pandas_dataset import DATA_DICT
from path_search.search_with_rule import find_all_paths
from util_dir.geometry import get_angle, normalize_angle

from path_search.search_with_rule_v2 import path_search_rule


def get_last_angle(pos: np.ndarray, cl_array: np.ndarray, yaw: float, last_pos: np.ndarray) -> float:
    cl_ls = LineString(cl_array)
    point = Point(pos)

    dist = cl_ls.project(point)
    project_point = cl_ls.interpolate(dist)

    if dist + 0.5 <= cl_ls.length:
        far_point = cl_ls.interpolate(dist + 0.5)
        direct_array = np.array([far_point.x, far_point.y]) - np.array([project_point.x, project_point.y])
    else:
        close_dist = dist - 0.5 if dist - 0.5 >= 0 else 0
        close_point = cl_ls.interpolate(close_dist)
        direct_array = np.array([project_point.x, project_point.y]) - np.array([close_point.x, close_point.y])

    traj_array = pos-last_pos

    if abs(traj_array[0]) < 1e-3 and abs(traj_array[1]) < 1e-3:
        lane_angle = get_angle(direct_array, np.array([1, 0]))
        res = normalize_angle(abs(lane_angle-yaw))
    else:
        res = normalize_angle(abs(get_angle(direct_array, traj_array)))

    # return normalize_angle(abs(lane_angle-yaw))
    return res


def check_speed_limit(track_xy_array: np.ndarray, speed_array: np.ndarray, hd_map: HDMap):
    for idx in range(track_xy_array.shape[0]):
        lane_list = hd_map.find_lanelet(pos=track_xy_array[idx])

        if len(lane_list) == 0:
            return False
        else:
            min_speed = 10e6

            for lane_id in lane_list:
                lane_obj = hd_map.id_lane_dict[lane_id]
                speed_limit = lane_obj.get_speed_limit()

                if speed_limit is None:
                    continue
                else:
                    if speed_limit < min_speed:
                        min_speed = speed_limit

            if speed_array[idx] > min_speed:
                return False

    return True


def check_traffic_rule(track_xy_array: np.ndarray, start_lane_list: List[int],
                       hd_map: HDMap, roundabout: bool = False):
    all_path_list = find_all_paths(lane_list=start_lane_list,
                                   hd_map=hd_map,
                                   roundabout=roundabout)

    lane_set = set()

    for path_list in all_path_list:
        for path in path_list:
            lane_set = lane_set | set(path)

    for idx in range(1, track_xy_array.shape[0]):
        lane_list = hd_map.find_lanelet(pos=track_xy_array[idx])

        if len(lane_list) == 0:
            return False
        else:
            common_set = lane_set & set(lane_list)
            if len(common_set) == 0:
                return False

    return True


def check_feasible(track_full_info: np.ndarray, hd_map: HDMap, roundabout: bool = False):
    track_xy_array = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
    yaw_array = track_full_info[:, [DATA_DICT["psi_rad"]]].astype("float")

    vel_array = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype("float")

    speed_array = np.hypot(vel_array[:, 0], vel_array[:, 1])

    start_lane_list = hd_map.find_lanelet(pos=track_xy_array[0, :])

    if len(start_lane_list) == 0:
        return False

    speed_res = check_speed_limit(track_xy_array=track_xy_array,
                                  speed_array=speed_array,
                                  hd_map=hd_map)

    if not speed_res:
        return False

    rule_res = check_traffic_rule(track_xy_array=track_xy_array,
                                     start_lane_list=start_lane_list,
                                     hd_map=hd_map,
                                     roundabout=roundabout)

    if not rule_res:
        return False

    return True


def check_traj_reasonable(case_df: pandas.DataFrame, track_full_info: np.ndarray, track_id: int, case_id: int,
                          roundabout: bool, hd_map: HDMap):
    track_xy_full = track_full_info[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
    track_yaw_full = track_full_info[:, [DATA_DICT["psi_rad"]]].astype(float)

    cl_list, _ = path_search_rule(track_obs_xy=track_xy_full[:10],
                                  track_obs_yaw=track_yaw_full[:10],
                                  case_data=case_df,
                                  track_id=track_id,
                                  hd_map=hd_map,
                                  roundabout=roundabout)

    for idx, cl in enumerate(cl_list):
        angle = get_last_angle(pos=track_xy_full[9],
                               cl_array=cl,
                               yaw=track_yaw_full[9],
                               last_pos=track_xy_full[8])
        if angle < math.pi/2:
            return True

    return False


def find_target_veh(map_path: str, data_path: str):
    data_df = pd.read_csv(data_path)

    hd_map = HDMap(osm_file_path=map_path)

    target_veh_dict = {}
    data_df = data_df[data_df["agent_type"] == "car"]
    case_set = data_df[["case_id"]].drop_duplicates(["case_id"]).values

    roundabout = True if "Roundabout" in map_path else False

    for i in tqdm(range(case_set.shape[0])):
        case_id = int(case_set[i])
        case_df = data_df[data_df["case_id"] == case_id]

        track_set = case_df[["track_id"]].drop_duplicates(["track_id"]).values

        for j in range(track_set.shape[0]):
            track_id = int(track_set[j])

            track_full_info = case_df[case_df["track_id"] == track_id].values

            if np.unique(track_full_info[:, [DATA_DICT["agent_type"]]]) != "car":
                continue

            vel_array = track_full_info[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype("float")

            if track_full_info.shape[0] != 40:
                continue

            if abs(vel_array[9][0]) < 1e-3 and abs(vel_array[9][1]) < 1e-3:
                continue

            if not check_traj_reasonable(case_df=case_df, track_full_info=track_full_info, track_id=track_id,
                                     case_id=case_id, roundabout=roundabout, hd_map=hd_map):
                continue

            feasible = check_feasible(track_full_info=track_full_info, hd_map=hd_map, roundabout=roundabout)

            if not feasible:
                continue
            else:
                if case_id in target_veh_dict.keys():
                    target_veh_dict[case_id].append(track_id)
                else:
                    target_veh_dict[case_id] = [track_id]

    return target_veh_dict


def save_res(basic_data_p, basic_map_p, basic_save_p, file, num_i):
    print(file)
    data_p = os.path.join(basic_data_p, file)
    map_p = os.path.join(basic_map_p, file[:-num_i] + ".osm")
    target_dict = find_target_veh(map_p, data_p)
    save_path = os.path.join(basic_save_p, file[:-num_i] + ".json")
    json_obj = json.dumps(target_dict)
    with open(save_path, "w") as f:
        f.write(json_obj)
        f.close()


if __name__ == '__main__':
    mode = "train"
    basic_data_p = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "/"
    basic_map_p = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps/"

    basic_save_p = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "_target_filter_delete_final"

    file_list = os.listdir(basic_data_p)

    # file_list = ["DR_DEU_Roundabout_OF_val.csv"]

    if mode == "val":
        num_i = 8
    else:
        num_i = 10

    Parallel(n_jobs=8)(delayed(save_res)(basic_data_p, basic_map_p, basic_save_p, file, num_i) for file in file_list)

    # for file in file_list:
    #     print(file)
    #     save_res(basic_data_p, basic_map_p, basic_save_p, file, num_i)
