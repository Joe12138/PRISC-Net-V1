import math
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm

sys.path.append("/home/joe/Desktop/TRCVTPP/RulePRIMEV2")
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from hdmap.hd_map import HDMap
# from test import get_target_vehicle
from path_generation.quintic_generation import quintic_polynomial_planner, quintic_generation
from util_dir.metric import get_ade, get_fde

from hdmap.visual.map_vis import draw_lanelet_map

def get_target_vehicle(file_path: str):
    key_dict = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        target_dict = json.load(f)

        for k in target_dict.keys():
            case_id = int(k)
            key_dict[case_id] = []
            for track_id in target_dict[k]:
                key_dict[case_id].append((case_id, track_id))
        # print(target_dict)
        f.close()
    return key_dict

mode = "val"

content = "yaw"

path_list = os.listdir("/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"+mode+"_target_filter")

if content == "speed":
    diff_list = [-50+i*5 for i in range(10)]
    diff_list.append(0)
    for i in range(1, 11):
        diff_list.append(i*5)

    print(diff_list)
    diff_dict = {}
    for i in diff_list:
        diff_dict[i] = []
else:
    diff_list = [-0.5*math.pi + i * 0.05 * math.pi for i in range(10)]
    diff_list.append(0)
    for i in range(1, 11):
        diff_list.append(i*0.05*math.pi)

    print(diff_list)
    diff_dict = {}
    for i in diff_list:
        diff_dict[i] = []

total_ade = []
total_fde = []

max_ade = -10e9
key = None

idx = 0

for file in path_list:
    file_name = file[:-5]

    print("---------------{}-------------------".format(file_name))

    map_path = "/home/joe/Desktop/PredictionWithIRL/maps/"+file_name+".osm"
    data_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "/"+file_name+"_"+mode+".csv"
    target_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "_target_filter/"+file_name+".json"

    dataset = DatasetPandas(data_path=data_path)
    hd_map = HDMap(osm_file_path=map_path)
    target_v = get_target_vehicle(file_path=target_path)

    for case_id in tqdm(target_v.keys()):
        case_data = dataset.get_case_data(case_id)
        for _, track_id in target_v[case_id]:
            track_full = dataset.get_track_data(case_id, track_id)

            gt_traj = track_full[10:, [DATA_DICT["x"], DATA_DICT["y"]]]

            start_state = track_full[9, :]
            end_state = track_full[-1, :]

            start_vx, start_vy = track_full[9, DATA_DICT["vx"]], track_full[9, DATA_DICT["vy"]]
            l_start_vx, l_start_vy = track_full[8, DATA_DICT["vx"]], track_full[9, DATA_DICT["vy"]]
            start_speed = math.sqrt(start_vx**2+start_vy**2)
            l_start_speed = math.sqrt(l_start_vx**2+l_start_vy**2)
            end_vx, end_vy = track_full[-1, DATA_DICT["vx"]], track_full[-1, DATA_DICT["vy"]]
            l_end_vx, l_end_vy = track_full[-2, DATA_DICT["vx"]], track_full[-1, DATA_DICT["vy"]]
            end_speed = math.sqrt(end_vx**2+end_vy**2)
            l_end_speed = math.sqrt(l_end_vx**2+l_end_vy**2)

            t_rx, t_ry = quintic_polynomial_planner(
                sx=start_state[DATA_DICT["x"]],
                sy=start_state[DATA_DICT["y"]],
                syaw=start_state[DATA_DICT["psi_rad"]],
                sv=start_speed,
                sa=(start_speed - l_start_speed) / 0.1,
                gx=end_state[DATA_DICT["x"]],
                gy=end_state[DATA_DICT["y"]],
                gyaw=end_state[DATA_DICT["psi_rad"]],
                gv=end_speed,  # end_speed,
                ga=(end_speed - l_end_speed) / 0.1
            )

            t_pred_traj = [(t_rx[i], t_ry[i]) for i in range(len(t_rx))]
            t_pred_traj_array = np.asarray(t_pred_traj)

            t_ade = get_ade(t_pred_traj_array, gt_traj)
            t_fde = get_fde(t_pred_traj_array, gt_traj)

            total_ade.append(t_ade)
            total_fde.append(t_fde)
            # idx += 1

            # test the effect of end_speed
            # if content == "speed":
            #     para_list = np.linspace(end_speed-5, end_speed+5, 21)
            # else:
            #     end_yaw = end_state[DATA_DICT["psi_rad"]]
            #     para_list = np.linspace(end_yaw-0.5*math.pi, end_yaw+0.5*math.pi, 21)
            # for idx, i in enumerate(para_list):
            #     rx, ry = quintic_polynomial_planner(
            #         sx=start_state[DATA_DICT["x"]],
            #         sy=start_state[DATA_DICT["y"]],
            #         syaw=start_state[DATA_DICT["psi_rad"]],
            #         sv=start_speed,
            #         sa=(start_speed-l_start_speed)/0.1,
            #         gx=end_state[DATA_DICT["x"]],
            #         gy=end_state[DATA_DICT["y"]],
            #         gyaw=end_state[DATA_DICT["psi_rad"]] if content == "speed" else i,
            #         gv=i if content == "speed" else end_speed,  # end_speed,
            #         ga=(end_speed-l_end_speed)/0.1
            #     )

            # prx, pry = quintic_generation(
            #     sx=start_state[DATA_DICT["x"]],
            #     sy=start_state[DATA_DICT["y"]],
            #     svx=start_vx,
            #     svy=start_vy,
            #     sax=(start_vx-l_start_vx)/0.1,
            #     say=(start_vy-l_start_vy)/0.1,
            #     gx=end_state[DATA_DICT["x"]],
            #     gy=end_state[DATA_DICT["y"]],
            #     gvx=end_vx,
            #     gvy=end_vy,
            #     gax=(end_vx-l_end_vx)/0.1,
            #     gay=(end_vy-l_end_vx)/0.1
            # )
            #     pred_traj = [(rx[i], ry[i]) for i in range(len(rx))]
            #     pred_traj_array = np.asarray(pred_traj)
            #
            #     ade = get_ade(pred_traj_array, gt_traj)
            #     # fde = get_fde(pred_traj_array, gt_traj)
            #
            #     # print("speed={}, real_speed={}, ade={}, fde={}".format(i, end_speed, ade, fde))
            #
            #     diff_dict[diff_list[idx]].append(abs(ade-t_ade))

                # total_ade += ade
                # total_fde += fde
                # idx += 1
                #
                # if ade > max_ade:
                #     max_ade = ade
                #     key = (case_id, track_id)
                # break
            # break
        # break
    # break
            #
            # axes = plt.subplot(111)
            # axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)
            # axes.scatter(rx, ry, color="purple", marker="o", s=25)
            # axes.scatter(prx, pry, color="green", marker="s", s=25)
            # axes.scatter(gt_traj[:, 0], gt_traj[:, 1], color="black", marker="*", s=25)
            # plt.show()

    # print("ade = {}\nfde = {}".format(total_ade/idx, total_fde/idx))
    # print("max_ade = {}, key = {}".format(max_ade, key))

    # break

# x = [i/10 for i in diff_list]
#
# mean_list = []
# yerr = np.zeros((2, len(x)))
#
# min_max_yerr = np.zeros((2, len(x)))
# for idx, diff in enumerate(diff_list):
#     mean_val = np.mean(diff_dict[diff])
#     var_val = np.var(diff_dict[diff])
#     # print(var_val)
#
#     min_val = np.min(diff_dict[diff])
#     max_val = np.max(diff_dict[diff])
#     mean_list.append(mean_val)
#
#     yerr[0, idx] = var_val
#     yerr[1, idx] = var_val
#
#     min_max_yerr[0, idx] = mean_val-min_val
#     min_max_yerr[1, idx] = max_val-mean_val
#
# plt.errorbar(x, mean_list, yerr=yerr[:, :], ecolor='k', elinewidth=0.5, marker='s', mfc='orange',
#              mec='k', mew=1, ms=5, alpha=1, capsize=5, capthick=3, linestyle="none", label="ADE")
#
# # plt.errorbar(x, mean_list, yerr=min_max_yerr[:, :], ecolor='b', elinewidth=0.5, marker='s', mfc='orange',
# #              mec='orange', mew=1, ms=2, alpha=1, capsize=5, capthick=3, linestyle="none")
# # plt.plot(x, mean_list, linestyle="-.", color="k")
# plt.xlabel(r"Difference of speed (m/s)", fontsize=10)
# plt.ylabel(r"ADE", fontsize=10)
# plt.xticks(fontsize=10)
# plt.yticks(fontsize=10)
# plt.legend(fontsize=10)
#
# plt.savefig(f"/home/joe/Desktop/yaw_error_{mode}.png", dpi=600)
# plt.close()

print("avg-ade = {}, acg-fde = {}".format(np.mean(total_ade), np.mean(total_fde)))
print("std-ade = {}, std-fde = {}".format(np.std(total_ade), np.std(total_fde)))
print("min-ade = {}, min-fde = {}".format(np.min(total_ade), np.min(total_fde)))
print("max-ade = {}, min-fde = {}".format(np.max(total_ade), np.max(total_fde)))