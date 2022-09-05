import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from hdmap.hd_map import HDMap
from compute_feature.compute_all_feature_rule import get_target_veh_list
from path_generation.frenet_generation import FrenetPlanner
from path_generation.object.vehicle_state import VehicleState
from path_generation.object.reference_line import ReferenceLine
from path_search.search_with_rule_v2 import path_search_rule, filter_cl_with_distance
from util_dir.metric import get_ade, get_fde
from path_generation.utils.discrete_points_utils import compute_path_profile
from path_generation.utils.generator_utils import convert_veh_frenet

from hdmap.visual.map_vis import draw_lanelet_map
from target_prediction.model.yaw_vel_inference import YawVelInference

mode = "val"
prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2"

yaw_model_path = "/home/joe/Desktop/Rule-PRIME/Model/yaw_output/06-27-18-51/best_YawVelPredict.pth"
dataset_path = f"/home/joe/ServerBackup/final_version_rule_equal_interval_0_25/{mode}_intermediate"
yaw_pred_inference = YawVelInference(dataset_path=dataset_path, model_path=yaw_model_path, variable="yaw")

target_path_prefix = f"{prefix}/{mode}_target_filter"
data_path_prefix = f"{prefix}/{mode}"
map_path_prefix = f"{prefix}/maps"

file_list = os.listdir(target_path_prefix)

total_ade = []
total_fde = []

# max_ade = -10e9
# key = None

idx = 0

for file in file_list:
    file_name = file[:-5]

    print("---------------{}-------------------".format(file_name))
    map_path = os.path.join(map_path_prefix, f"{file_name}.osm")
    data_path = os.path.join(data_path_prefix, f"{file_name}_{mode}.csv")
    # target_path = os.path.join(target_path_prefix, f"{file_name}.json")

    dataset = DatasetPandas(data_path=data_path)
    hd_map = HDMap(osm_file_path=map_path)
    target_v = get_target_veh_list(target_veh_path=target_path_prefix, file_name=file)

    for scene_name, case_id, track_id in tqdm(target_v):
        # if (scene_name, case_id, track_id) != ("DR_USA_Roundabout_EP", 321, 1):
        #     continue
        yaw_target, _ = yaw_pred_inference.inference(scene_name=scene_name,
                                                     case_id=case_id,
                                                     track_id=track_id)

        yaw_pred = yaw_target[0][0]
        case_data = dataset.get_case_data(case_id)
        track_full = dataset.get_track_data(case_id, track_id)
        track_xy_full = track_full[:, [DATA_DICT["x"], DATA_DICT["y"]]]
        track_yaw_full = track_full[:, [DATA_DICT["psi_rad"]]]
        heading, accumulated_s, kappas, dkappas = compute_path_profile(track_xy_full)
        track_obs_full = track_full[:10, :]
        vx = track_full[:, DATA_DICT["vx"]]
        vy = track_full[:, DATA_DICT["vy"]]
        track_vel = [np.hypot(vx[i], vy[i]) for i in range(40)]

        cl_list, _ = path_search_rule(track_obs_xy=track_xy_full[:10],
                                   track_obs_yaw=track_yaw_full[:10],
                                   case_data=case_data,
                                   track_id=track_id,
                                   hd_map=hd_map,
                                   roundabout=True if "Roundabout" in file_name else False)

        start_veh_state = VehicleState(
            x=track_obs_full[-1, DATA_DICT["x"]],
            y=track_obs_full[-1, DATA_DICT["y"]],
            vx=track_obs_full[-1, DATA_DICT["vx"]],
            vy=track_obs_full[-1, DATA_DICT["vy"]],
            acc=(track_vel[10] - track_vel[8]) / 0.2,
            yaw=track_obs_full[-1, DATA_DICT["psi_rad"]],
            kappa=kappas[9],
            width=track_obs_full[-1, DATA_DICT["width"]],
            length=track_obs_full[-1, DATA_DICT["length"]]
        )

        end_veh_state = VehicleState(
            x=track_full[-1, DATA_DICT["x"]],
            y=track_full[-1, DATA_DICT["y"]],
            vx=track_full[-1, DATA_DICT["vx"]],
            vy=track_full[-1, DATA_DICT["vy"]],
            acc=(track_vel[-1] - track_vel[-2]) / 0.1,
            yaw=track_full[-1, DATA_DICT["psi_rad"]],
            kappa=kappas[-1],
            width=track_full[-1, DATA_DICT["width"]],
            length=track_full[-1, DATA_DICT["length"]]
        )

        # ade_list = []
        # fde_list = []
        cl_copy_list = []
        for cl in cl_list:
            # print(cl)
            cl_no_repeat = []
            for i in range(cl.shape[0]):
                if i == 0:
                    cl_no_repeat.append((cl[i][0], cl[i][1]))
                elif i == cl.shape[0]-1:
                    cl_no_repeat.append((cl[i][0], cl[i][1]))
                else:
                    if abs(cl[i+1][0]-cl[i][0]) < 1e-6 or abs(cl[i+1][1]-cl[i][1]) < 1e-6:
                        continue
                    else:
                        cl_no_repeat.append((cl[i][0], cl[i][1]))
            cl_copy = np.asarray(cl_no_repeat)
            cl_copy_list.append(cl_copy)

        min_cl_idx = filter_cl_with_distance(
            cl_idx_list=[i for i in range(len(cl_copy_list))],
            track_obs_array=track_xy_full,
            cl_list=cl_copy_list
        )
        cl = cl_copy_list[min_cl_idx]
        try:
            ref_line = ReferenceLine(waypoint_x=cl[:, 0], waypoint_y=cl[:, 1], wps_step=0.1)

            frenet_traj = FrenetPlanner(veh_state=start_veh_state, plan_time=3, plan_dt=0.1)
            frenet_traj.init_by_target_lane(refer_line=ref_line)

            # End state
            x, y, yaw, kappa, vel, acc = end_veh_state.get_global_info()

            s, d, rx, ry, ryaw, rkappa, rkappa_prime = ref_line.get_correspond_rpoint(x, y)
            # print("s = {}, d = {}".format(s, d))

        
            s_d, s_dd, d_d, d_dd, d_prime, d_prime_prime = convert_veh_frenet(yaw, kappa, vel, acc, d, ryaw, rkappa,
                                                                            rkappa_prime)
            traj = frenet_traj.trajectory_generation(
                lon_vel=s_d,
                lon_acc=s_dd,
                lat_d=d,
                lat_vec=d_d,
                lat_acc=d_dd
            )

            traj_fut_array = np.concatenate((np.asarray(traj.x).reshape(-1, 1), (np.asarray(traj.y).reshape(-1, 1))), axis=1)

            

            # print(traj_fut_array)
            ade = get_ade(forecasted_trajectory=traj_fut_array, gt_trajectory=track_xy_full[10:])
            fde = get_fde(forecasted_trajectory=traj_fut_array, gt_trajectory=track_xy_full[10:])

            if ade > 0:
                print(scene_name, case_id, track_id)
                axes = plt.subplot(111)
                axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)
                axes.plot(traj.x, traj.y, color="purple", marker="o", markersize=5, zorder=10)
                axes.plot(cl[:, 0], cl[:, 1], color="gray", linestyle="--")

                axes.plot(ref_line.rx, ref_line.ry, color="green", linestyle="-.")
                axes.scatter(track_xy_full[:, 0], track_xy_full[:, 1], color="red", marker="*", s=5)
                plt.show()
            # # ade_list.append(ade)
            # # fde_list.append(fde)
            # # print(ade, fde)
            # if ade > 10000 or fde > 10000:
            #     continue
            total_ade.append(ade)
            total_fde.append(fde)
        except RuntimeError:
            continue
    
        # idx += 1
    # break

print("avg-ade = {}, acg-fde = {}".format(np.mean(total_ade), np.mean(total_fde)))
print("std-ade = {}, std-fde = {}".format(np.std(total_ade), np.std(total_fde)))
print("min-ade = {}, min-fde = {}".format(np.min(total_ade), np.min(total_fde)))
print("max-ade = {}, min-fde = {}".format(np.max(total_ade), np.max(total_fde)))