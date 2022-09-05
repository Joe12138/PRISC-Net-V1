import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
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

from typing import List, Dict, Tuple, Any, Optional, Set
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from shapely.geometry import LineString, Point

from hdmap.hd_map import HDMap
from dataset.pandas_dataset import DATA_DICT, DatasetPandas
from path_search.search_with_rule_v2 import path_search_rule
from path_generation.object.reference_line import ReferenceLine

from compute_feature.utils.baseline_config import _LANE_GAUSSIAN_FILTER, _LANE_GAUSSIAN_SIGMA, \
    _FILTER_NBR_NEAREST_DIST, _FILTER_NBR_STATIONARY, _PADDING_COLUMN_FLAG
from compute_feature.utils.map_features_utils import get_oracle_centerline
from compute_feature.utils.statistic_utils import calcu_preditions_error, PredsErrorStat, save_csv_stat
from compute_feature.utils.social_features_utils import SocialFeaturesUtils
from path_generation.object.config import _WAYPOINTS_STEP
from path_generation.utils.math_utils import cal_rot_matrix

from target_prediction.model.target_inference import TargetPredInference
from target_prediction.model.yaw_vel_inference import YawVelInference
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


def check_off_road(traj: np.ndarray, hd_map: HDMap):
    for i in range(traj.shape[0]):
        lanelet_list = hd_map.find_lanelet(traj[i])

        if len(lanelet_list) == 0:
            return False

    return True

def get_speed(traj: np.ndarray):
    speed_list = []
    
    for i in range(traj.shape[0]):
        if i == 0:
            p_1 = traj[i]
            p_2 = traj[i+1]
            diff = p_2-p_1
            speed = math.sqrt(diff[0]**2+diff[1]**2)/0.1
        elif i == 29:
            p_1 = traj[i-1]
            p_2 = traj[i]
            diff = p_2-p_1
            speed = math.sqrt(diff[0]**2+diff[1]**2)/0.1
        else:
            p_1 = traj[i-1]
            p_2 = traj[i+1]
            diff = p_2-p_1
            speed = math.sqrt(diff[0]**2+diff[1]**2)/0.2
        speed_list.append(speed)
    return speed_list


def get_speed_limit(hd_map:HDMap, lane_list: List[int]):
    speed_limit = None
    
    for lane in lane_list:
        lane_obj = hd_map.id_lane_dict[lane]
        cur_sl = lane_obj.get_speed_limit()
        if cur_sl is None:
            continue
        else:
            if speed_limit is None:
                speed_limit = cur_sl
            else:
                if speed_limit > cur_sl:
                    speed_limit = cur_sl
    return speed_limit

def check_violate_rule(hd_map: HDMap, traj_array: np.ndarray, lane_set: Set[int]):
    speed_list = get_speed(traj=traj_array)
    for i in range(traj_array.shape[0]):
        pos = traj_array[i, :]
        lane_list = hd_map.find_lanelet(pos)
        
        if len(lane_list) == 0:
            return 1
        else:
            if len(set(lane_list) & lane_set) == 0:
                return 2
            else:
                speed_limit = get_speed_limit(hd_map=hd_map, lane_list=lane_list)
                if speed_limit is None:
                        continue
                else:
                    if speed_list[i] > speed_limit+1:
                        # print(speed_list[j], speed_limit)
                        return 3
    return 0

def get_candidate_trajs(scene_name: str,
                        case_id: int,
                        track_id: int,
                        target_pred_inference: TargetPredInference,
                        lanes: List[ReferenceLine],
                        hd_map: HDMap,
                        init_veh_state: Dict[str, float],
                        lane_set: Set[int],
                        yaw_pred_inference: Optional[YawVelInference] = None):
    futs_xy_for_lanes = []
    futs_sd_for_lanes = []

    for _ in lanes:
        futs_xy_for_lanes.append([])
        futs_sd_for_lanes.append([])

    pred_target = target_pred_inference.inference(scene_name=scene_name,
                                                  case_id=case_id,
                                                  track_id=track_id)

    # yaw_target, _ = yaw_pred_inference.inference(scene_name=scene_name,
    #                                              case_id=case_id,
    #                                              track_id=track_id)

    # a = yaw_target[0]

    for i in range(20):
        closest_lane, lane_idx, target_p, lane_yaw = get_closest_lane(target_p=pred_target[i], lanes=lanes, hd_map=hd_map)

        # for j in range(1):
        lane_list = hd_map.find_lanelet(pos=target_p[0])
        speed_limit = get_speed_limit(hd_map=hd_map, lane_list=lane_list)
        
        max_speed = init_veh_state["v"]+3 if speed_limit is None else abs(speed_limit)
        min_speed = max(0, init_veh_state["v"]-3)
        traj_state_dict = {0: [], 1: [], 2: [], 3: []}
        for speed in np.arange(min_speed, max_speed+0.3, 0.3):
            # for a in np.arange(-1, 1+0.2, 0.2):
            a = 0
            try:
                rx, ry = quintic_polynomial_planner(sx=init_veh_state["x"],
                                                    sy=init_veh_state["y"],
                                                    syaw=init_veh_state["yaw"],
                                                    sv=init_veh_state["v"],
                                                    sa=init_veh_state["a"],
                                                    gx=target_p[0][0],
                                                    gy=target_p[0][1],
                                                    gyaw=lane_yaw,
                                                    gv=speed,
                                                    ga=a)
            except Exception:
                continue
            pred_traj_array = np.array([rx, ry]).transpose()
            result = check_violate_rule(hd_map=hd_map, traj_array=pred_traj_array, lane_set=lane_set)
            traj_state_dict[result].append(pred_traj_array)
                # if check_off_road(traj=pred_traj_array, hd_map=hd_map):
                #     futs_xy_for_lanes[lane_idx].append(pred_traj_array)
                #     futs_sd_for_lanes[lane_idx].append(closest_lane.get_track_sd(pred_traj_array))
        if len(traj_state_dict[0]) != 0:
            dist_dict = {}
            for idx, traj_array in enumerate(traj_state_dict[0]):
                traj_ls = LineString(traj_array)
                dist_dict[idx] = traj_ls.length
            
            dist_list = sorted(dist_dict.items(), key=lambda item: item[1])
            for i in range(min(len(traj_state_dict[0]), 4)):
                index = dist_list[i][0]
                pred_traj_array = traj_state_dict[0][index]
                futs_xy_for_lanes[lane_idx].append(pred_traj_array)
                futs_sd_for_lanes[lane_idx].append(closest_lane.get_track_sd(pred_traj_array))
        else:
            if len(traj_state_dict[3]) != 0:
                traj_length = 10e9
                pred_traj_array = None
                for traj_array in traj_state_dict[3]:
                    traj_ls = LineString(traj_array)
                    if traj_length > traj_ls.length:
                        traj_length = traj_ls.length
                        pred_traj_array = copy.deepcopy(traj_array)
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
                            target_pred_inference: TargetPredInference,
                            yaw_pred_inference: YawVelInference):
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
    agent_centerlines, agent_path = path_search_rule(track_obs_xy=track_obs_xy,
                                         track_obs_yaw=track_obs_yaw,
                                         case_data=case_data,
                                         track_id=track_id,
                                         hd_map=hd_map,
                                         roundabout=roundabout)
    
    path_set = set()
    for path in agent_path:
        for lane_id in path:
            path_set.add(lane_id)
            
    if len(agent_centerlines) == 0:
        logging.error(f"No centerlines generated from {seq[0]}_{seq[1]}_{seq[2]}")
        return None, None, None, None

    processed_lanes = get_processed_lanes(centerlines=agent_centerlines)

    # ======================== 2. Generate potential trajectories ===========================
    # wait for a good solution
    # print("gt_target = {}".format(track_gt_xy[-1]))
    init_veh_state = get_init_veh_state(track_full_info=agent_track_full)
    agent_futs_xy, agent_futs_sd = get_candidate_trajs(scene_name=scene_name,
                                                       case_id=case_id,
                                                       track_id=track_id,
                                                       target_pred_inference=target_pred_inference,
                                                       lanes=processed_lanes,
                                                       hd_map=hd_map,
                                                       lane_set=path_set,
                                                       init_veh_state=init_veh_state,
                                                       yaw_pred_inference=yaw_pred_inference)

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
                           target_pred_inference: TargetPredInference,
                           yaw_pred_inference:YawVelInference):
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
                                    target_pred_inference=target_pred_inference,
                                    yaw_pred_inference=yaw_pred_inference)

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
    prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    mode = "val"
    data_path = prefix + f"{mode}"
    map_path = prefix + "maps"
    target_veh_path = prefix + f"{mode}_target_filter"
    save_path = f"/home/joe/Desktop/Rule-PRIME/Data/feature_test/frenet_{mode}_v2"
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

    model_path = "/home/joe/Desktop/trained_model/new_target_pred/best_TargetPredict.pth"
    yaw_model_path = "/home/joe/Desktop/Rule-PRIME/Model/yaw_output/06-27-18-51/best_YawVelPredict.pth"
    dataset_path = f"/home/joe/ServerBackup/final_version_rule_equal_interval_0_25/{args.mode}_intermediate"

    target_pred_inference = TargetPredInference(dataset_path=dataset_path, model_path=model_path)
    yaw_pred_inference = YawVelInference(dataset_path=dataset_path, model_path=yaw_model_path, variable="yaw")
    print("Load inference successfully!")
    social_features_utils_instance = SocialFeaturesUtils()

    file_name = "DR_CHN_Roundabout_LN.json"

    load_seq_save_features(save_dir=pkl_save_dir,
                            file_name=file_name,
                            social_features_utils_instance=social_features_utils_instance,
                            target_pred_inference=target_pred_inference,
                            yaw_pred_inference=yaw_pred_inference)


    # file_list = os.listdir(args.target_vehicle_dir)

    # for file_name in file_list:
    #     load_seq_save_features(save_dir=pkl_save_dir,
    #                         file_name=file_name,
    #                         social_features_utils_instance=social_features_utils_instance,
    #                         target_pred_inference=target_pred_inference,
    #                            yaw_pred_inference=yaw_pred_inference)

    

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

    print(f"Prediction computation for {args.data_dir} set completed in {(time.time() - start_time) / 60.0} mins")