import os
import json
import numpy as np
import math

from hdmap.hd_map import HDMap
from compute_feature.compute_all_feature_rule import get_candidate_trajs, get_processed_lanes
from target_prediction.model.target_inference import TargetPredInference
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from path_search.search_with_rule import path_search_rule

from util_dir.metric import get_ade, get_fde
from tqdm import tqdm
import csv

def get_key_list(target_veh_dir: str):
    key_list = []

    file_list = os.listdir(target_veh_dir)

    for file in file_list:
        scene_name = file[:-5]
        with open(os.path.join(target_veh_dir, file), "r", encoding="UTF-8") as f:
            target_dict = json.load(f)

            for k in target_dict.keys():
                case_id = int(k)

                for track_id in target_dict[k]:
                    key_list.append((scene_name, case_id, track_id))

            f.close()
    return key_list

def get_map_dict(map_dir: str):
    map_dict = {}

    map_file_list = os.listdir(map_dir)

    for file in map_file_list:
        if "xy" in file:
            continue
        print(file)
        scene_name = file[:-4]
        hd_map = HDMap(osm_file_path=os.path.join(map_dir, file))

        map_dict[scene_name] = hd_map

    return map_dict


def get_data_dict(mode: str, data_dir: str):
    data_dict = {}

    if mode == "train":
        num_i = 10
    elif mode == "val":
        num_i = 8
    else:
        raise Exception("No this split model: {}".format(mode))

    data_file_list = os.listdir(data_dir)

    for file in data_file_list:
        scene_name = file[:-num_i]
        data_pandas = DatasetPandas(data_path=os.path.join(data_dir, file))

        data_dict[scene_name] = data_pandas

    return data_dict

if __name__ == '__main__':
    mode = "val"
    target_veh_path = f"/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/{mode}_target_filter/"

    map_path = f"/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps/"
    data_path = f"/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/{mode}/"

    model_path = "/home/joe/Desktop/06-05-06-41/best_TargetPredict.pth"
    dataset_path = f"/home/joe/Dataset/rule/{mode}_intermediate"

    key_list = get_key_list(target_veh_path)

    map_dict = get_map_dict(map_dir=map_path)
    data_dict = get_data_dict(mode=mode, data_dir=data_path)

    target_pred_inference = TargetPredInference(dataset_path=dataset_path, model_path=model_path)

    obs_len, seq_len = 10, 40

    min_ade_list = []
    min_fde_list = []

    f = open(f"/home/joe/Desktop/{mode}_pred.csv", "w", encoding="UTF-8")

    csv_writer = csv.writer(f)

    number = 10
    head_list = ["scene_name", "case_id", "track_id"]

    for i in range(number):
        head_list.append(f"fde_{i+1}")
        head_list.append(f"ade_{i+1}")

    head_list.append("min_ade")
    head_list.append("min_fde")

    csv_writer.writerow(head_list)

    for scene_name, case_id, track_id in tqdm(key_list):
        roundabout = True if "Roundabout" in scene_name else False

        case_data = data_dict[scene_name].get_case_data(case_id=case_id)
        agent_track = case_data[case_data["track_id"] == track_id].values

        # If track is of OBS_LEN (test mode), use agent_track of full SEQ_LEN,
        # But keep (OBS_LEN+1) to (SEQ_LEN) indexes having None values.
        if agent_track.shape[0] == obs_len:
            agent_track_full = np.full((seq_len, agent_track.shape[1]), None)
            agent_track_full[:obs_len] = agent_track
            track_gt_xy = None
        else:
            agent_track_full = agent_track
            track_gt_xy = agent_track_full[obs_len:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")

        track_obs = agent_track_full[:obs_len]
        track_obs_xy = agent_track_full[:obs_len, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")

        track_obs_yaw = track_obs[:, [DATA_DICT["psi_rad"]]].astype("float")

        ######################### Agent's feature #############################
        # ======================== 1. Get potential centerlines ===========================
        agent_centerlines = path_search_rule(track_obs_xy=track_obs_xy,
                                             track_obs_heading=track_obs_yaw,
                                             case_data=case_data,
                                             track_id=track_id,
                                             hd_map=map_dict[scene_name],
                                             roundabout=roundabout)
        if len(agent_centerlines) == 0:
            print(f"No centerlines generated from {scene_name}_{case_id}_{track_id}")

        processed_lanes = get_processed_lanes(centerlines=agent_centerlines)

        start_vx, start_vy = agent_track_full[9, DATA_DICT["vx"]], agent_track_full[9, DATA_DICT["vy"]]
        l_start_vx, l_start_vy = agent_track_full[8, DATA_DICT["vx"]], agent_track_full[9, DATA_DICT["vy"]]
        start_speed = math.sqrt(start_vx ** 2 + start_vy ** 2)
        l_start_speed = math.sqrt(l_start_vx ** 2 + l_start_vy ** 2)

        # sx = start_state[DATA_DICT["x"]],
        # sy = start_state[DATA_DICT["y"]],
        # syaw = start_state[DATA_DICT["psi_rad"]],
        # sv = start_speed,
        # sa = (start_speed - l_start_speed) / 0.1,
        init_veh_dict = {"x": track_obs_xy[-1][0],
                         "y": track_obs_xy[-1][1],
                         "yaw": track_obs[-1][DATA_DICT["psi_rad"]],
                         "v": start_speed,
                         "a": (start_speed-l_start_speed)/0.1}
        fut_xy_traj, _ = get_candidate_trajs(scene_name=scene_name,
                                             case_id=case_id,
                                             track_id=track_id,
                                             target_pred_inference=target_pred_inference,
                                             lanes=processed_lanes,
                                             hd_map=map_dict[scene_name],
                                             init_veh_state=init_veh_dict)
        ade_list = []
        fde_list = []
        for traj_list in fut_xy_traj:
            for traj in traj_list:
                ade = get_ade(traj, track_gt_xy)
                fde = get_fde(traj, track_gt_xy)

                ade_list.append(ade)
                fde_list.append(fde)

        data_list = [scene_name, case_id, track_id]

        for fde, ade in zip(ade_list, fde_list):
            data_list.append(fde)
            data_list.append(ade)

        min_ade = np.min(ade_list)
        min_fde = np.min(fde_list)

        data_list.append(min_ade)
        data_list.append(min_fde)

        csv_writer.writerow(data_list)

        min_ade_list.append(min_ade)
        min_fde_list.append(min_fde)

    f.close()

    print(f"min_ade = {np.mean(min_ade_list)}, min_fde = {np.mean(min_fde_list)}")

