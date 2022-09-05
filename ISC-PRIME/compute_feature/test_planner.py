import os
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import numpy as np
from tqdm import tqdm
from hdmap.hd_map import HDMap
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from compute_all_feature_rule import get_target_veh_list, get_candidate_trajs, get_init_veh_state, get_processed_lanes
from target_prediction.model.target_inference import TargetPredInference

from path_search.search_with_rule_v2 import path_search_rule
from util_dir.metric import get_ade, get_fde

import matplotlib.pyplot as plt
from hdmap.visual.map_vis import draw_lanelet_map


mode = "val"
data_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2"
map_path = os.path.join(data_prefix, "maps")
data_path = os.path.join(data_prefix, mode)
target_path = os.path.join(data_prefix, f"{mode}_target_filter")

# target pred model
# /home/joe/Desktop/trained_model/more_vel_yaw_64/checkpoint_iter246.ckpt
target_pred_model_path = "/home/joe/Desktop/trained_model/more_vel_yaw_64/checkpoint_iter246.ckpt"
dataset_path = f"/home/joe/ServerBackup/final_version_rule_equal_interval_0_25/{mode}_intermediate"

target_pred_inference = TargetPredInference(dataset_path=dataset_path, model_path=target_pred_model_path)

target_file_list = os.listdir(target_path)

obs_len = 10

ade_list = []
ade_fde_list = []

fde_list = []
fde_ade_list = []

for file_name in target_file_list:
    scene_name = file_name[:-5]
    print(scene_name)
    # if scene_name != "DR_CHN_Merging_ZS2":
    #     continue

    hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))
    data_pandas = DatasetPandas(data_path=os.path.join(data_path, f"{scene_name}_{mode}.csv"))
    target_veh_list = get_target_veh_list(target_veh_path=target_path, file_name=file_name)

    for _, case_id, track_id in tqdm(target_veh_list):
        # if (case_id, track_id) != (64, 6):
        #     continue
        case_data = data_pandas.get_case_data(case_id=case_id)
        agent_track = case_data[case_data["track_id"] == track_id].values

        track_obs = agent_track[:obs_len]
        track_obs_xy = agent_track[:obs_len, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
        track_gt_xy = agent_track[obs_len:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")

        # print(track_obs)

        track_obs_yaw = track_obs[:, [DATA_DICT["psi_rad"]]].astype("float")

        agent_centerlines, _ = path_search_rule(track_obs_xy=track_obs_xy,
                                                track_obs_yaw=track_obs_yaw,
                                                case_data=case_data,
                                                track_id=track_id,
                                                hd_map=hd_map,
                                                roundabout=True if "Roundabout" in scene_name else False)

        processed_lanes = get_processed_lanes(centerlines=agent_centerlines)

        init_veh_state = get_init_veh_state(track_full_info=agent_track)
        agent_futs_xy, agent_futs_sd = get_candidate_trajs(scene_name=scene_name,
                                                           case_id=case_id,
                                                           track_id=track_id,
                                                           target_pred_inference=target_pred_inference,
                                                           lanes=processed_lanes,
                                                           hd_map=hd_map,
                                                           init_veh_state=init_veh_state,
                                                           yaw_pred_inference=None)

        min_ade = 10e9
        min_ade_fde = -1
        min_fde = 10e9
        min_fde_ade = -1

        for xy_list in agent_futs_xy:
            # xy = np.asarray(xy)
            for xy in xy_list:
                ade = get_ade(xy, track_gt_xy)
                fde = get_fde(xy, track_gt_xy)

                if ade < min_ade:
                    min_ade = ade
                    min_ade_fde = fde

                if fde < min_fde:
                    min_fde = fde
                    min_fde_ade = ade

        if min_ade > 100 or min_fde > 50:
            # axes = plt.subplot(111)
            # axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)

            print("scene_name = {}, case_id = {}, track_id = {}".format(scene_name, case_id, track_id))

        ade_list.append(min_ade)
        ade_fde_list.append(min_ade_fde)

        fde_list.append(min_fde)
        fde_ade_list.append(min_fde_ade)
    break

# For ade
print("min ade = {}, corresponding fde = {}".format(np.mean(ade_list), np.mean(ade_fde_list)))

# For fde
print("min fde = {}, corresponding ade = {}".format(np.mean(fde_list), np.mean(fde_ade_list)))





