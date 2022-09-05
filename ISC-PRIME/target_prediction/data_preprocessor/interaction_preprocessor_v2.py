import argparse
import copy
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("/home/joelan/Desktop/Rule-PRIME/RulePRIME")
import os
import pandas as pd
from typing import Tuple, Dict, List
from shapely.geometry import LineString, Point

from target_prediction.data_preprocessor.preprocessor import Preprocessor
from hdmap.hd_map import HDMap
from hdmap.util.map_util import get_lane_id_in_xy_bbox, get_polygon
from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from util_dir.geometry import get_angle, normalize_angle
from path_search.search_with_rule import path_search_rule
from target_prediction.utils.cubic_spline import Spline2D
from hdmap.visual.map_vis import draw_lanelet_map, plot_lane_list
from target_prediction.data_preprocessor.interaction_preprocessor import get_lane_direction


def resample_cl_with_dist(cl_list: List[np.ndarray], dist: float = 1):
    resample_cl_list = []

    for cl in cl_list:
        cl_ls = LineString(cl)
        dist_list = np.arange(0, cl_ls.length, dist)
        cl_coord_list = []

        for dist_value in dist_list:
            point = cl_ls.interpolate(dist_value)
            cl_coord_list.append([point.x, point.y])

        resample_cl_list.append(np.asarray(cl_coord_list))

    return resample_cl_list


class InteractionPreprocessor(Preprocessor):
    def __init__(self,
                 data_path: str,
                 scene_name: str,
                 target_veh_path: str,
                 map_path: str,
                 split: str = "train",
                 obs_horizon: int = 10,
                 start_obs_lat_range: int = -5,
                 end_obs_lat_range: int = 15,
                 start_obs_lon_range: int = -5,
                 end_obs_lon_range: int = 5,
                 pred_horizon: int = 30,
                 normalized: bool = True,
                 save_dir=None
                 ):
        super(InteractionPreprocessor, self).__init__(root_dir=data_path)
        self.start_obs_lat_range = start_obs_lat_range
        self.end_obs_lat_range = end_obs_lat_range
        self.start_obs_lon_range = start_obs_lon_range
        self.end_obs_lon_range = end_obs_lon_range

        self.split = split
        self.normalized = normalized
        self.save_dir = save_dir

        self.scene_name = scene_name
        self.hd_map = HDMap(osm_file_path=map_path)
        self.data_pandas = DatasetPandas(data_path=data_path)
        self.target_veh_dict = self.get_target_veh_dict(target_veh_path=target_veh_path)

        self.case_list = list(self.target_veh_dict.keys())

        self.mapping = dict()

        # store all information
        self.step_dict = dict()
        self.traj_dict = dict()
        self.full_info_dict = dict()

        # store graph information
        # self.lane_dict = dict()

    def init_information_dict(self):
        self.step_dict.clear()
        self.traj_dict.clear()
        self.full_info_dict.clear()

    @staticmethod
    def get_target_veh_dict(target_veh_path: str) -> Dict[str, List[int]]:
        with open(target_veh_path, "r", encoding="UTF-8") as f:
            target_veh_dict = json.load(f)

        return target_veh_dict

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        case_id_str = self.case_list[item]
        case_id_int = int(case_id_str)

        case_data = copy.deepcopy(self.data_pandas.get_case_data(case_id=case_id_int))
        case_data.reset_index(drop=True, inplace=True)
        return self.process_target(case_data, case_id_int, list(self.target_veh_dict[case_id_str]))

    def process_target(self, case_data: pd.DataFrame, case_id: int, track_id_list: List[int]):
        # print(case_id)
        # if case_id == 65:
        #     print("here")
        self.read_interaction_data(case_data=case_data)

        for track_id in track_id_list:
            data = self.get_obj_feats(case_data=case_data, track_id=track_id)
            data["graph"] = self.get_lane_graph(data=data)

            file_name = f"{self.scene_name}_{case_id}_{track_id}"
            data["seq_id"] = file_name

            df_processed = pd.DataFrame(
                [[data[key] for key in data.keys()]],
                columns=[key for key in data.keys()]
            )

            self.save(dataframe=df_processed,
                      file_name=file_name,
                      dir_=self.save_dir)

        return []

    def read_interaction_data(self, case_data: pd.DataFrame):
        self.init_information_dict()

        agt_ts = np.sort(np.unique(case_data["frame_id"].values))

        # self.mapping.clear()
        if len(self.mapping) == 0:
            for i, ts in enumerate(agt_ts):
                self.mapping[ts] = i

        trajs = np.concatenate(
            (
                case_data.x.to_numpy().reshape(-1, 1),
                case_data.y.to_numpy().reshape(-1, 1)
            ), 1
        )

        full_info = np.concatenate(
            (
                case_data.x.to_numpy().reshape(-1, 1),
                case_data.y.to_numpy().reshape(-1, 1),
                case_data.vx.to_numpy().reshape(-1, 1),
                case_data.vy.to_numpy().reshape(-1, 1),
                case_data.psi_rad.to_numpy().reshape(-1, 1)
            ), 1
        )

        steps = [self.mapping[x] for x in case_data["frame_id"].values]
        steps = np.asarray(steps, int)

        objs = case_data.groupby(["track_id"]).groups
        keys = list(objs.keys())

        for key in keys:
            idcs = objs[key]
            self.traj_dict[key] = trajs[idcs]
            self.full_info_dict[key] = full_info[idcs]
            self.step_dict[key] = steps[idcs]

    def get_obj_feats(self, case_data: pd.DataFrame, track_id: int):
        orig = self.traj_dict[track_id][self.obs_horizon-1].copy().astype(float)
        obs_track_full = self.full_info_dict[track_id][:self.obs_horizon].copy().astype(float)

        roundabout = True if "Roundabout" in self.scene_name else False

        if self.normalized:
            theta = get_lane_direction(his_track_full=obs_track_full, hd_map=self.hd_map)
            rot = np.asarray(
                [
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]
                ], float
            )
        else:
            theta = None

            rot = np.asarray(
                [
                    [1.0, 0.0],
                    [0.0, 1.0]
                ], float
            )

        # get the target candidate and candidate gt
        agt_traj_obs = self.traj_dict[track_id][:self.obs_horizon].copy().astype(float)
        agt_traj_fut = self.traj_dict[track_id][self.obs_horizon: self.obs_horizon+self.pred_horizon].copy().astype(float)

        veh_yaw = obs_track_full[:, [4]].astype(float)

        ctr_line_candts = path_search_rule(track_obs_xy=agt_traj_obs,
                                           track_obs_heading=veh_yaw,
                                           case_data=case_data,
                                           track_id=track_id,
                                           hd_map=self.hd_map,
                                           roundabout=roundabout)

        ctr_line_candts = resample_cl_with_dist(cl_list=ctr_line_candts)

        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut-orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i]-orig.reshape(-1, 2)).T).T

        tar_candts = self.lane_candidate_sampling(centerline_list=ctr_line_candts,
                                                  distance=0.5,
                                                  viz=False)

        if self.split == "test":
            tar_candts_gt, tar_offset_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            # Why this will use the information of future trajectory?
            splines, ref_idx = self.get_ref_centerline(cline_list=ctr_line_candts, pred_gt=agt_traj_fut)
            tar_candts_gt, tar_offset_gt = self.get_candidate_gt(target_candidate=tar_candts, gt_target=agt_traj_fut[-1])

        x_min, x_max, y_min, y_max = self.start_obs_lat_range, self.end_obs_lat_range, self.start_obs_lon_range, self.end_obs_lon_range

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
        for track_idx in self.traj_dict.keys():
            if track_idx == track_id:
                continue

            traj = copy.deepcopy(self.traj_dict[track_idx])
            step = copy.deepcopy(self.step_dict[track_idx])

            if self.obs_horizon - 1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), float)
            has_pred = np.zeros(self.pred_horizon, bool)

            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask] - self.obs_horizon
            post_traj = traj_nd[future_mask]
            gt_pred[post_step] = post_traj
            has_pred[post_step] = True

            # collect the observation
            obs_mask = step < self.obs_horizon
            step_obs = step[obs_mask]
            traj_obs = traj_nd[obs_mask]
            idcs = step_obs.argsort()
            step_obs = step_obs[idcs]
            traj_obs = traj_obs[idcs]

            i = 0
            for i in range(len(step_obs)):
                if step_obs[i] == self.obs_horizon - len(step_obs) + i:
                    break
            step_obs = step_obs[i:]
            traj_obs = traj_obs[i:]

            if len(step_obs) <= 1:
                continue

            feat = np.zeros((self.obs_horizon, 3), float)
            has_obs = np.zeros(self.obs_horizon, bool)

            feat[step_obs, :2] = traj_obs
            feat[step_obs, 2] = 1
            has_obs[step_obs] = True

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue

            feats.append(feat)  # displacement vectors
            has_obss.append(has_obs)
            gt_preds.append(gt_pred)
            has_preds.append(has_pred)

        feats = np.asarray(feats, float)
        has_obss = np.asarray(has_obss, bool)
        gt_preds = np.asarray(gt_preds, float)
        has_preds = np.asarray(has_preds, bool)

        data = dict()
        data["track_id"] = track_id
        data["traj_dict"] = self.traj_dict
        data["full_info_dict"] = self.full_info_dict
        data["step_dict"] = self.step_dict
        data['orig'] = orig
        data['theta'] = theta
        data['rot'] = rot

        data['feats'] = feats
        data['has_obss'] = has_obss

        data['has_preds'] = has_preds
        data['gt_preds'] = gt_preds
        data['tar_candts'] = tar_candts
        data['gt_candts'] = tar_candts_gt
        data['gt_tar_offset'] = tar_offset_gt

        data['ref_ctr_lines'] = splines  # the reference candidate centerlines Spline for prediction
        data['ref_ctr_idx'] = ref_idx  # the idx of the closest reference centerlines

        return data

    def get_lane_graph(self, data: Dict):
        x_min, x_max, y_min, y_max = self.start_obs_lat_range, self.end_obs_lat_range, self.start_obs_lon_range, self.end_obs_lon_range
        radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))

        lane_ids = get_lane_id_in_xy_bbox(query_x=data["orig"][0],
                                          query_y=data["orig"][1],
                                          hd_map=self.hd_map,
                                          query_search_range_manhattan=radius)
        lane_ids = copy.deepcopy(lane_ids)

        lanes = dict()
        for lane_id in lane_ids:
            lane = self.hd_map.id_lane_dict[lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data["rot"], (lane.centerline_array - data["orig"].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]

            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = get_polygon(lane_obj=lane)
                polygon = copy.deepcopy(polygon)

                lane.centerline = centerline
                lane.polygon = np.matmul(data["rot"], (polygon[:, :2] - data["orig"].reshape(-1, 2)).T).T
                lanes[lane_id] = lane

        lane_ids = list(lanes.keys())
        # self.plot_graph(hd_map, lane_ids, data["orig"])
        ctrs, feats, turn, control, intersect, speed_limits = [], [], [], [], [], []

        for lane_id in lane_ids:
            lane = lanes[lane_id]
            ctrln = lane.centerline
            num_segs = len(ctrln) - 1

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

            has_control, turn_direction, is_intersect, speed_limit = self.hd_map.get_lane_info(lane_id=lane_id)
            x = np.zeros((num_segs, 2), np.float32)
            if turn_direction == 'LEFT':
                x[:, 0] = 1
            elif turn_direction == 'RIGHT':
                x[:, 1] = 1
            else:
                pass
            turn.append(x)
            control.append(has_control * np.ones(num_segs, np.float32))
            intersect.append(is_intersect * np.ones(num_segs, np.float32))
            speed_limits.append(speed_limit * np.ones(num_segs, np.float32))

        lane_idcs = []
        count = 0
        # node 计算方式感觉有点问题
        for i, ctr in enumerate(ctrs):
            lane_idcs.append(i * np.ones(len(ctr), np.int64))
            count += len(ctr)

        num_nodes = count
        lane_idcs = np.concatenate(lane_idcs, 0)

        graph = dict()
        graph['ctrs'] = np.concatenate(ctrs, 0)
        graph['num_nodes'] = num_nodes
        graph['feats'] = np.concatenate(feats, 0)
        graph['turn'] = np.concatenate(turn, 0)
        graph['control'] = np.concatenate(control, 0)
        graph['intersect'] = np.concatenate(intersect, 0)
        graph['lane_idcs'] = lane_idcs

        return graph

    @staticmethod
    def get_ref_centerline(cline_list: List[np.ndarray], pred_gt: np.ndarray):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        elif len(cline_list) == 0:
            raise Exception("No centerline.")
        else:
            line_idx = 0
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1])
                               for i in range(len(cline_list))]

            # search the closest point of the traj final position to each center line
            min_distances = []
            for line in ref_centerlines:
                xy = np.stack([line.x_fine, line.y_fine], axis=1)
                diff = xy - pred_gt[-1, :2]
                dis = np.hypot(diff[:, 0], diff[:, 1])
                min_distances.append(np.min(dis))
            line_idx = np.argmin(min_distances)
            return ref_centerlines, line_idx

    @staticmethod
    def get_ref_centerline_v2(cline_list: List[np.ndarray], pred_gt: np.ndarray):
        if len(cline_list) == 1:
            return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
        else:
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

            min_distance = 10e9
            line_idx = None

            for idx, line in enumerate(cline_list):
                cl_ls = LineString(coordinates=line)

                project_p_list = []
                for i in range(pred_gt.shape[0]):
                    gt_point = Point(pred_gt[i])
                    dist = cl_ls.project(gt_point)
                    project_p = cl_ls.interpolate(dist)
                    project_p_list.append((project_p.x, project_p.y))

                project_p_array = np.asarray(project_p_list)
                diff = pred_gt - project_p_array
                nl_dist = np.hypot(diff[:, 0], diff[:, 1])
                if nl_dist < min_distance:
                    line_idx = idx

            return ref_centerlines, line_idx


def main_func(data_path: str, target_veh_path: str, map_path: str, scene_name: str, split: str, save_dir: str):
    print("hello_{}".format(scene_name))
    preprocessor = InteractionPreprocessor(data_path=data_path,
                                           scene_name=scene_name,
                                           target_veh_path=target_veh_path,
                                           map_path=map_path,
                                           split=split,
                                           save_dir=save_dir)

    loader = DataLoader(preprocessor,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False
                        )

    for i, data in enumerate(tqdm(loader)):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_name", type=str, default="DR_USA_Intersection_EP0.json")
    parser.add_argument("-m", "--mode", type=str, default="val")

    args = parser.parse_args()

    mode = args.mode
    file = args.file_name
    path_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"
    data_path = path_prefix + mode
    target_path = path_prefix + mode + "_target_filter/"
    map_path = path_prefix + "/maps/"
    save_path = "/home/joe/Dataset/original_target_data"
    file_list = os.listdir(target_path)

    # str = ""
    # for idx, file in enumerate(file_list):
    #     if idx == 0:
    #         str += f"python interaction_preprocessor_v2.py -m {mode} -f {file}"
    #     else:
    #         str += f" & python interaction_preprocessor_v2.py -m {mode} -f {file}"
    # print(str)
    # print(file_list)
    # Parallel(n_jobs=11)(
    #     delayed(main_func)(
    #         os.path.join(data_path, f"{file[:-5]}_{mode}.csv"),
    #         os.path.join(target_path, file),
    #         os.path.join(map_path, f"{file[:-5]}.osm"),
    #         file[:-5],
    #         mode,
    #         save_path
    #     )
    #     for file in file_list
    # )

    os.makedirs(os.path.join(save_path, f"{mode}_intermediate", "raw"), exist_ok=True)
    main_func(data_path=os.path.join(data_path, f"{file[:-5]}_{mode}.csv"),
              target_veh_path=os.path.join(target_path, file),
              map_path=os.path.join(map_path, f"{file[:-5]}.osm"),
              scene_name=file[:-5],
              split=mode,
              save_dir=save_path)



