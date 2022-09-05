import os
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

from shapely.geometry import LineString, Point
from tqdm import tqdm

from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from compute_feature.compute_all_feature_rule import get_target_veh_list
from hdmap.hd_map import HDMap
from path_search.search_with_rule_v2 import path_search_rule
from util_dir.geometry import get_angle, normalize_angle

from hdmap.visual.map_vis import draw_lanelet_map

### 分析车辆最后朝向与真实那条道路中心线的角度距离

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


def get_norm_distance(gt_traj: np.ndarray, cl_array: np.ndarray) -> float:
    cl_ls = LineString(cl_array)

    project_points = []

    for i in range(gt_traj.shape[0]):
        point = Point(gt_traj[i])
        dist = cl_ls.project(point)
        project_point = cl_ls.interpolate(dist)
        project_points.append((project_point.x, project_point.y))

    diff_array = project_points-gt_traj

    result = np.hypot(diff_array[:, 0], diff_array[:, 1])

    return np.sum(result)


mode = "val"
all_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
map_path = os.path.join(all_prefix, "maps")
data_path = os.path.join(all_prefix, mode)
target_veh_path = os.path.join(all_prefix, f"{mode}_target_filter_delete_final")

file_list = os.listdir(target_veh_path)

angle_list = []

for file_name in file_list:
    scene_name = file_name[:-5]
    print(scene_name)
    target_veh_list = get_target_veh_list(target_veh_path=target_veh_path, file_name=file_name)
    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))

    for _, case_id, track_id in tqdm(target_veh_list):
        # if (case_id, track_id) != (307, 7):
        #     continue
        case_df = data_pandas.get_case_data(case_id=case_id)
        track_info_full = case_df[case_df["track_id"] == track_id].values

        track_xy_full = track_info_full[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype(float)
        track_yaw_full = track_info_full[:, [DATA_DICT["psi_rad"]]].astype(float)

        cl_list, _ = path_search_rule(track_obs_xy=track_xy_full[:10],
                                      track_obs_yaw=track_yaw_full[:10],
                                      case_data=case_df,
                                      track_id=track_id,
                                      hd_map=hd_map,
                                      roundabout=True if "Roundabout" in scene_name else False)
        min_dist = 10e9
        real_angle = 10e9
        cl_idx = -1
        for idx, cl in enumerate(cl_list):
            angle = get_last_angle(pos=track_xy_full[9],
                                   cl_array=cl,
                                   yaw=track_yaw_full[9],
                                   last_pos=track_xy_full[8])
            dist = get_norm_distance(gt_traj=track_xy_full,
                                     cl_array=cl)

            if dist < min_dist:
                min_dist = dist
                real_angle = angle
                cl_idx = idx

        if abs(real_angle) > math.pi/2:
            print(scene_name, case_id, track_id)
            print(real_angle)
            axes = plt.subplot(111)
            axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)

            for idx, cl in enumerate(cl_list):
                if idx == cl_idx:
                    axes.plot(cl[:, 0], cl[:, 1], color="purple", zorder=10)
                else:
                    axes.plot(cl[:, 0], cl[:, 1], color="gray")
            axes.plot(track_xy_full[:10, 0], track_xy_full[:10, 1], color="red")
            axes.scatter(track_xy_full[9, 0], track_xy_full[9, 1], color="red", marker="o")
            axes.plot(track_xy_full[10:, 0], track_xy_full[10:, 1], color="blue")
            axes.scatter(track_xy_full[-1, 0], track_xy_full[-1, 1], color="blue", marker="o")

            plt.show()
            plt.cla()
            plt.clf()

        angle_list.append(real_angle)

    break

print("mean = {}, var = {}, min = {}, max = {}".format(np.mean(angle_list), np.std(angle_list), np.min(angle_list),
                                                       np.max(angle_list)))
