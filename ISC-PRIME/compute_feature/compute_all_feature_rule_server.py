import sys
sys.path.append("/home/ubuntu/traj_pred_project/Rule-TRCVTP/TRCVTPP/RulePRIMEV2")
import copy
import logging
import math
import os
import time
import argparse
import pathlib
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple, Any
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point

from hdmap.hd_map import HDMap
from dataset.pandas_dataset import DATA_DICT, DatasetPandas
from path_search.search_with_rule import path_search_rule
from path_generation.object.reference_line import ReferenceLine

from compute_feature.utils.baseline_config import _LANE_GAUSSIAN_FILTER, _LANE_GAUSSIAN_SIGMA, \
    _FILTER_NBR_NEAREST_DIST, _FILTER_NBR_STATIONARY, _PADDING_COLUMN_FLAG
from compute_feature.utils.map_features_utils import get_oracle_centerline
from compute_feature.utils.statistic_utils import calcu_preditions_error, PredsErrorStat, save_csv_stat
from compute_feature.utils.social_features_utils import SocialFeaturesUtils
from path_generation.object.config import _WAYPOINTS_STEP
from path_generation.utils.math_utils import cal_rot_matrix

from target_prediction.dataloader.graph_data import GraphData

from target_prediction.model.target_inference import TargetPredInference
from util_dir.geometry import get_angle, normalize_angle

from path_generation.quintic_generation import quintic_polynomial_planner

from joblib import Parallel, delayed


def get_processed_lanes(centerlines: List[np.ndarray]) -> List[ReferenceLine]:
    lanes = []

    for cl in centerlines:
        if _LANE_GAUSSIAN_FILTER:
            waypoint_x = gaussian_filter(cl[:, 0], sigma=_LANE_GAUSSIAN_SIGMA)
            waypoint_y = gaussian_filter(cl[:, 1], sigma=_LANE_GAUSSIAN_SIGMA)
        else:
            waypoint_x, waypoint_y = cl[:, 0], cl[:, 1]
        lanes.append(ReferenceLine(waypoint_x=waypoint_x,
                                   waypoint_y=waypoint_y,
                                   wps_step=_WAYPOINTS_STEP))

    return lanes


def get_closest_lane(target_p: np.ndarray, lanes: List[ReferenceLine], hd_map: HDMap):
    min_dist = 10e9
    closest_lane = None
    lane_idx = None
    closest_point = None
    yaw = None

    for idx, lane in enumerate(lanes):
        target_point = Point(target_p[0])
        project_dist = lane.ref_line_ls.project(target_point)
        closest_p = lane.ref_line_ls.interpolate(project_dist)

        dist = np.linalg.norm(target_p-np.array([closest_p.x, closest_p.y]))

        if dist < min_dist:
            min_dist = dist
            closest_lane = copy.deepcopy(lane)
            lane_idx = idx
            closest_point = np.array([closest_p.x, closest_p.y])

            if project_dist != lane.ref_line_ls.length:
                if project_dist+1 > lane.ref_line_ls.length:
                    end_p = lane.ref_line_ls.interpolate(lane.ref_line_ls.length)
                else:
                    end_p = lane.ref_line_ls.interpolate(project_dist+1)
                direction_array = np.array([end_p.x, end_p.y])-np.array([closest_p.x, closest_p.y])
            else:
                if project_dist-1 < 0:
                    start_p = lane.ref_line_ls.interpolate(0)
                else:
                    start_p = lane.ref_line_ls.interpolate(project_dist-1)
                direction_array = np.array([closest_p.x, closest_p.y])-np.array([start_p.x, start_p.y])

            yaw = normalize_angle(get_angle(vec_a=np.array([1, 0]), vec_b=direction_array))

    lane_list = hd_map.find_lanelet(pos=target_p[0])
    if len(lane_list) == 0:
        ls = LineString(np.vstack((closest_point, target_p)))
        split_factor = 0.5

        while True:
            interpolate_p = ls.interpolate(split_factor, normalized=True)
            lane_list = hd_map.find_lanelet(pos=np.array([interpolate_p.x, interpolate_p.y]))
            if len(lane_list) == 0:
                split_factor /= 2
            else:
                target_p = np.array([[interpolate_p.x, interpolate_p.y]])
                break

    return closest_lane, lane_idx, target_p, yaw


def get_candidate_trajs(scene_name: str,
                        case_id: int,
                        track_id: int,
                        target_pred_inference: TargetPredInference,
                        lanes: List[ReferenceLine],
                        hd_map: HDMap,
                        init_veh_state: Dict[str, float]):
    futs_xy_for_lanes = []
    futs_sd_for_lanes = []

    for _ in lanes:
        futs_xy_for_lanes.append([])
        futs_sd_for_lanes.append([])

    pred_target = target_pred_inference.inference(scene_name=scene_name,
                                                  case_id=case_id,
                                                  track_id=track_id)

    for i in range(10):
        closest_lane, lane_idx, target_p, lane_yaw = get_closest_lane(target_p=pred_target[i], lanes=lanes, hd_map=hd_map)

        rx, ry = quintic_polynomial_planner(sx=init_veh_state["x"],
                                            sy=init_veh_state["y"],
                                            syaw=init_veh_state["yaw"],
                                            sv=init_veh_state["v"],
                                            sa=init_veh_state["a"],
                                            gx=target_p[0][0],
                                            gy=target_p[0][1],
                                            gyaw=lane_yaw,
                                            gv=init_veh_state["v"],
                                            ga=0)

        pred_traj_array = np.array([rx, ry]).transpose()
        futs_xy_for_lanes[lane_idx].append(pred_traj_array)
        futs_sd_for_lanes[lane_idx].append(closest_lane.get_track_sd(pred_traj_array))

    return futs_xy_for_lanes, futs_sd_for_lanes


def get_lanes_projection(lanes: List[ReferenceLine], xy: np.ndarray) -> List[np.ndarray]:
    sd_list = []
    for lane in lanes:
        sd = lane.get_track_sd(xy)
        sd_list.append(sd)
    return sd_list


def get_social_features(df: pd.DataFrame,
                        track_id: int,
                        obs_len: int,
                        lanes: List[ReferenceLine],
                        social_features_utils_instance: SocialFeaturesUtils,
                        data_format: Dict[str, int],
                        viz: bool = False,
                        filter_nearest_dist=None,
                        filter_stationary_tracks=False,
                        padding_column_flag=True) -> Tuple[List[np.ndarray], List[List[np.ndarray]]]:
    """Compute features from neighbour vehicles
    Args:
        track: df: dataframe
    """
    agent_track = df[df["track_id"] == track_id].values
    agent_obs_xy = agent_track[:obs_len, [data_format["x"], data_format["y"]]].astype("float")
    social_tracks_obs = social_features_utils_instance.get_social_tracks(df, track_id, obs_len, data_format,
                                                                         filter_stationary_tracks,
                                                                         padding_column_flag=padding_column_flag)
    nbrs_hist_xy = []
    nbr_lanes_sd = []
    for nbr_track_obs in social_tracks_obs:
        # Obtain the xy coordinate of each nbr
        # padding_column: 0 for Origin data, 1 for Padding data
        if padding_column_flag:
            hist_xy = nbr_track_obs[:obs_len, [DATA_DICT["x"], DATA_DICT["y"], 6]].astype("float")
        else:
            hist_xy = nbr_track_obs[:obs_len, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")

        if filter_nearest_dist and \
                np.sqrt(np.sum((hist_xy[:,:2] - agent_obs_xy)**2, axis=1)).min() > filter_nearest_dist:
            continue
        nbrs_hist_xy.append(hist_xy)

        hist_sds = get_lanes_projection(lanes, hist_xy[:, :2])
        if padding_column_flag:
            for idx in range(len(hist_sds)):
                hist_sds[idx] = np.hstack((hist_sds[idx], hist_xy[:,2].reshape(obs_len,1)))
        nbr_lanes_sd.append(hist_sds)

    # Turn to the fomat of lane_num * nbr_num
    nbrs_hist_sd_on_lanes = [ [nbr[lane_id] for nbr in nbr_lanes_sd] for lane_id in range(len(lanes)) ]

    # Visualization
    if viz:
        plt.figure(figsize=(8, 7))
        for idx, nbr_traj in enumerate(nbrs_hist_xy):
            plt.plot(nbr_traj[:, 0], nbr_traj[:, 1], "--", color="dimgrey", alpha=1, linewidth=1, zorder=5)
            plt.plot(nbr_traj[-1, 0], nbr_traj[-1, 1], "o", color="red", alpha=1, markersize=7, zorder=6)
            plt.text(nbr_traj[-1, 0], nbr_traj[-1, 1], f"{idx}")
        plt.plot(agent_obs_xy[:, 0], agent_obs_xy[:, 1], "-", color="#ECA154", alpha=1, linewidth=1, zorder=10)
        plt.plot(agent_obs_xy[-1, 0], agent_obs_xy[-1, 1], "o", color="#ECA154", alpha=1, markersize=7, zorder=11)
        agent_traj_fut = agent_track[obs_len:, [data_format["X"], data_format["Y"]]].astype("float")
        plt.plot(agent_traj_fut[:, 0], agent_traj_fut[:, 1], "-", color="#d33e4c", alpha=1, linewidth=1, zorder=15)
        plt.plot(agent_traj_fut[-1, 0], agent_traj_fut[-1, 1], "o", color="#d33e4c", alpha=1, markersize=7, zorder=16)
        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("equal")
        plt.title("Number of social nbrs = {}".format(len(nbrs_hist_xy)))
        plt.show()

    return nbrs_hist_xy, nbrs_hist_sd_on_lanes


def get_init_veh_state(track_full_info: np.ndarray):
    l_s_vx, l_s_vy = track_full_info[8, DATA_DICT["vx"]], track_full_info[8, DATA_DICT["vy"]]
    s_vx, s_vy = track_full_info[9, DATA_DICT["vx"]], track_full_info[9, DATA_DICT["vy"]]

    l_s_speed = math.sqrt(l_s_vx**2+l_s_vy**2)
    s_speed = math.sqrt(s_vx**2+s_vy**2)

    init_veh_state = {
        "x": track_full_info[9, DATA_DICT["x"]],
        "y": track_full_info[9, DATA_DICT["y"]],
        "yaw": track_full_info[9, DATA_DICT["psi_rad"]],
        "v": s_speed,
        "a": (s_speed-l_s_speed)/0.1,
    }
    return init_veh_state


def compute_frenet_features(case_data: pd.DataFrame,
                            scene_name: str,
                            case_id: int,
                            track_id: int,
                            hd_map: HDMap,
                            roundabout: bool,
                            obs_len: int,
                            pred_len: int,
                            social_features_utils_instance: SocialFeaturesUtils,
                            seq: Tuple[str, int, int],
                            target_pred_inference: TargetPredInference):
    seq_len = obs_len + pred_len
    viz = False

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
                                         hd_map=hd_map,
                                         roundabout=roundabout)
    if len(agent_centerlines) == 0:
        logging.error(f"No centerlines generated from {seq[0]}_{seq[1]}_{seq[2]}")
        return None, None, None, None

    processed_lanes = get_processed_lanes(centerlines=agent_centerlines)

    # ======================== 2. Generate potential trajectories ===========================
    # wait for a good solution
    init_veh_state = get_init_veh_state(track_full_info=agent_track_full)
    agent_futs_xy, agent_futs_sd = get_candidate_trajs(scene_name=scene_name,
                                                       case_id=case_id,
                                                       track_id=track_id,
                                                       target_pred_inference=target_pred_inference,
                                                       lanes=processed_lanes,
                                                       hd_map=hd_map,
                                                       init_veh_state=init_veh_state)

    if all([trajs == [] for trajs in agent_futs_xy]):
        logging.error(f"No trajectories generated from {seq[0]}_{seq[1]}_{seq[2]}!!!")
        return None, None, None, None

    # ======================== 3. Fuse centerlines by trajs result ===========================
    feasible_cls_id = np.array([id for id in range(len(agent_centerlines)) if len(agent_futs_xy[id]) > 0])

    # MARK: filter the following things with no feasible trajectories
    agent_centerlines = [agent_centerlines[id] for id in feasible_cls_id]
    agent_futs_xy = [agent_futs_xy[id] for id in feasible_cls_id]
    agent_futs_sd = [agent_futs_sd[id] for id in feasible_cls_id]

    # MARK: process the class of Lane for the following projection
    processed_lanes = [processed_lanes[id] for id in feasible_cls_id]
    agent_obs_sds = get_lanes_projection(processed_lanes, track_obs_xy)

    # MARK: find the oracle one after the cls-trajs-fusion
    agent_oracle_clid, probability_of_cls = get_oracle_centerline(track_gt_xy, agent_centerlines)

    # Compute for csv file generating : prediction errors and vehicle state information
    trajs_error_stat = calcu_preditions_error(agent_futs_xy, track_gt_xy)
    vel = np.sqrt(track_obs[-1, DATA_DICT["vx"]] ** 2 + track_obs[-1, DATA_DICT["vy"]] ** 2)
    vel_last = np.sqrt(track_obs[-2, DATA_DICT["vx"]] ** 2 + track_obs[-2, DATA_DICT["vy"]] ** 2)
    trajs_error_stat.add_veh_stat(veh_stat=[vel, (vel-vel_last)/0.1, track_obs[-1, DATA_DICT["psi_rad"]]])

    agent_cls_feature = (agent_centerlines, agent_oracle_clid)  # ================================ AGENT_CLS_FEATURE
    agent_trajs_feature = (agent_track_full, agent_obs_sds, agent_futs_sd, agent_futs_xy)  # ======== AGENT_TRAJS_FEATURE

    ################################# Neighbors' feature #################################
    ################################# Neighbors' feature #################################
    nbrs_hist_xy, nbrs_hist_sd_on_lanes = get_social_features(case_data, track_id, obs_len,
                                                              processed_lanes,
                                                              social_features_utils_instance,
                                                              DATA_DICT,
                                                              viz=viz,
                                                              filter_nearest_dist=_FILTER_NBR_NEAREST_DIST,
                                                              filter_stationary_tracks=_FILTER_NBR_STATIONARY,
                                                              padding_column_flag=_PADDING_COLUMN_FLAG)

    nbrs_trajs_feature = (nbrs_hist_xy, nbrs_hist_sd_on_lanes)  # =============================== NBRS_TRAJS_FEATURES

    return agent_cls_feature, agent_trajs_feature, nbrs_trajs_feature, trajs_error_stat


def get_target_veh_list(target_veh_path: str, file_name: str):
    target_veh_list = []
    scene_name = file_name[:-5]
    with open(os.path.join(target_veh_path, file_name), "r", encoding="UTF-8") as f:
        target_dict = json.load(f)

        for k in target_dict.keys():
            case_id = int(k)

            for track_id in target_dict[k]:
                target_veh_list.append((scene_name, case_id, track_id))

        f.close()
    return target_veh_list


def load_seq_save_features(save_dir: str,
                           file_name: str,
                           social_features_utils_instance: SocialFeaturesUtils,
                           target_pred_inference: TargetPredInference):
    args = parse_arguments()

    logging.basicConfig(filename=f"{os.path.abspath(args.feature_dir)}/logging/record_{args.mode}.log",
                        filemode="w",
                        level=logging.ERROR,
                        format="[%(levelname)s]%(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    os.makedirs(save_dir, exist_ok=True)

    hd_map = HDMap(osm_file_path=os.path.join(args.map_dir, f"{file_name[:-5]}.osm"))
    target_veh_list = get_target_veh_list(target_veh_path=args.target_vehicle_dir,
                                          file_name=file_name)
    dataset_pandas = DatasetPandas(data_path=os.path.join(args.data_dir, f"{file_name[:-5]}_{args.mode}.csv"))

    for scene_name, case_id, track_id in tqdm(target_veh_list):
        logging.info(f"--------------------Sequence{scene_name}_{case_id}_{track_id}--------------------------")
        case_data = dataset_pandas.get_case_data(case_id=case_id)
        preds_valid = True
        sample_valid = True
        ## ============================== Compute social and map features ==============================
        agent_cls_feature, agent_trajs_feature, nbrs_trajs_feature, trajs_error_stat = \
            compute_frenet_features(case_data=case_data,
                                    scene_name=scene_name,
                                    case_id=case_id,
                                    track_id=track_id,
                                    hd_map=hd_map,
                                    roundabout=True if "Roundabout" in file_name else False,
                                    obs_len=args.obs_len,
                                    pred_len=args.pred_len,
                                    social_features_utils_instance=social_features_utils_instance,
                                    seq=(scene_name, case_id, track_id),
                                    target_pred_inference=target_pred_inference)

        seq = f"{scene_name}_{case_id}_{track_id}"
        ## ========================================== Save CSV ==========================================
        # MARK: #### Jump the df without generating all features ( No centerlines found / No trajs generated )
        if (agent_cls_feature is None) or (agent_trajs_feature is None) or (nbrs_trajs_feature is None):
            logging.error(f"{seq}.csv: Pass feature computation")
            trajs_error_stat = PredsErrorStat()
            preds_valid = False
        trajs_error_stat.add_seq_id(seq=seq)
        save_csv_stat(first_row=(seq == f"{target_veh_list[0][0]}_{target_veh_list[0][1]}_{target_veh_list[0][2]}"),
                      save_loc=f"{os.path.abspath(args.feature_dir)}/stat/{scene_name}.csv", stat=trajs_error_stat)
        rot_matrix = cal_rot_matrix(trajs_error_stat.yaw)

        ## ========================================== Save data ==========================================
        # MARK: #### Jump the df which fde > 10.0 in train dataset only
        if trajs_error_stat.min_fde >= 10.0 and args.mode == 'train':
            logging.error(f"{seq}.csv -- BIG FDE!.")
            sample_valid = False

        # if trajs_error_stat.trajs_num <= 50.0:
        #     logging.error(f"{seq}.csv -- Trajs num less than 50")

        if preds_valid and sample_valid:
            oracle_cl_ids = [] if (agent_cls_feature[-1] is None) else agent_cls_feature[-1]
            traj_nums_list = [len(i) for i in agent_trajs_feature[-1]]
            traj_num_stat = ''.join(str(num) + '*' * (i in oracle_cl_ids) + '/' for i, num in enumerate(traj_nums_list))

            # Mark: Save this df ==========================================
            df = pd.DataFrame(data=[[seq, agent_cls_feature, agent_trajs_feature, nbrs_trajs_feature, rot_matrix]],
                              columns=["SEQUENCE", "AGENT_CLS_FEATURE", "AGENT_TRAJS_FEATURE", "NBRS_TRAJS_FEATURES",
                                       "YAW_MAT"])
            # Save each computed feature for a sequences as a single file
            df.to_pickle(os.path.join(save_dir, f"seq_{seq}.pkl"))


def create_dirs(root_dir: str):
    """Save all the things into feature_dir"""
    root_dir = pathlib.Path(root_dir)
    root_dir.mkdir(exist_ok=True)

    os.makedirs(f"{os.path.abspath(root_dir)}/split", exist_ok=False)
    os.makedirs(f"{os.path.abspath(root_dir)}/logging", exist_ok=False)
    os.makedirs(f"{os.path.abspath(root_dir)}/stat", exist_ok=False)


def parse_arguments() -> Any:
    """Parse command line arguments"""
    prefix = "/home/ubuntu/traj_pred_project/INTERACTION-Dataset-DR-single-v1_2/"
    mode = "test"
    data_path = prefix + f"{mode}"
    map_path = prefix + "maps"
    target_veh_path = prefix + f"{mode}_target_filter"
    save_path = f"/home/ubuntu/traj_pred_project/Rule-TRCVTP/TRCVTPP/RulePRIMEV2/compute_feature/feature/frenet_{mode}/"
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=data_path, help="The directory where the vehicle data save.")
    parser.add_argument("--map_dir", type=str, default=map_path, help="The directory where the map data save.")
    parser.add_argument("--target_vehicle_dir", type=str, default=target_veh_path,
                        help="The directory where the target vehicle save.")

    parser.add_argument("--mode", type=str, default=mode, help="/dataset_type: train/val/test")
    parser.add_argument("--feature_dir", type=str, default=save_path,
                        help="The directory where the computed features and related things are to be saved.")

    parser.add_argument("--num_seqs", type=int, default=8, help="Threads number for parallel computation.")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for parallel computation.")

    parser.add_argument("--obs_len", type=int, default=10, help="Observed length of the trajectory.")
    parser.add_argument("--pred_len", type=int, default=30, help="Prediction Horizon")

    return parser.parse_args()


if __name__ == '__main__':
    """Load sequences and save the computed features"""
    start_time = time.time()
    args = parse_arguments()
    create_dirs(args.feature_dir)

    pkl_save_dir = os.path.abspath(args.feature_dir)+"/split/"

    model_path = "/home/ubuntu/traj_pred_project/Rule-TRCVTP/06-05-06-41/best_TargetPredict.pth"
    dataset_path = f"/home/ubuntu/traj_pred_project/dataset/rule/{args.mode}_intermediate"

    target_pred_inference = TargetPredInference(dataset_path=dataset_path, model_path=model_path)

    print("Loading inference model successful!")
    social_features_utils_instance = SocialFeaturesUtils()

    # file_name = "DR_CHN_Roundabout_LN.json"

    file_list = os.listdir(args.target_vehicle_dir)

    for file_name in file_list:
        load_seq_save_features(save_dir=pkl_save_dir,
                            file_name=file_name,
                            social_features_utils_instance=social_features_utils_instance,
                            target_pred_inference=target_pred_inference)

    # Try to use parallel
    # Parallel(n_jobs=len(file_list))(
    #     delayed(load_seq_save_features)(
    #         pkl_save_dir,
    #         file_name,
    #         social_features_utils_instance,
    #         target_pred_inference
    #     )
    #     for file_name in file_list
    # )

    # print(f"Prediction computation for {args.data_dir} set completed in {(time.time() - start_time) / 60.0} mins")