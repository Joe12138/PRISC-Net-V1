import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME/")
import os
import copy
import json
import argparse

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Dict, List
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from shapely.geometry import LineString, Point

from dataset.pandas_dataset import DatasetPandas
from hdmap.hd_map import HDMap
from hdmap.util.map_util import get_lane_id_in_xy_bbox, get_polygon

from target_prediction.data_preprocessor.interaction_preprocessor import get_lane_direction
from target_prediction.data_preprocessor.interaction_preprocessor_v2 import resample_cl_with_dist
from target_prediction.utils.cubic_spline import Spline2D
from path_search.search_with_rule_v2 import path_search_rule


class InteractionPreprocessor(Dataset):
    def __init__(self,
                 data_path: str,
                 scene_name: str,
                 target_veh_path: str,
                 map_path: str,
                 split: str = "train",
                 obs_horizon: int = 10,
                 pred_horizon: int = 30,
                 start_obs_lat_range: int = -10,
                 end_obs_lat_range: int = 50,
                 start_obs_lon_range: int = -10,
                 end_obs_lon_range: int = 10,
                 normalized: bool = True,
                 save_dir=None):
        super(InteractionPreprocessor, self).__init__()

        self.start_obs_lat_range = start_obs_lat_range
        self.end_obs_lat_range = end_obs_lat_range

        self.start_obs_lon_range = start_obs_lon_range
        self.end_obs_lon_range = end_obs_lon_range

        self.split = split
        self.normalized = normalized
        self.save_dir = save_dir

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon

        self.hd_map = HDMap(osm_file_path=map_path)
        self.data_pandas = DatasetPandas(data_path=data_path)
        self.target_veh_dict = self.get_target_veh_dict(target_veh_path=target_veh_path)

        self.scene_name = scene_name

        self.case_list = list(self.target_veh_dict.keys())

        self.mapping = dict()

        # store all information
        self.step_dict = dict()
        self.traj_dict = dict()
        self.full_info_dict = dict()

    @staticmethod
    def get_target_veh_dict(target_veh_path: str) -> Dict[str, List[int]]:
        with open(target_veh_path, "r", encoding="UTF-8") as f:
            target_veh_dict = json.load(f)

        return target_veh_dict

    def init_information_dict(self):
        self.step_dict.clear()
        self.traj_dict.clear()
        self.full_info_dict.clear()

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        case_id_str = self.case_list[item]
        case_id_int = int(case_id_str)

        case_data = copy.deepcopy(self.data_pandas.get_case_data(case_id=case_id_int))
        # Reset the index of pandas.Dataframe
        case_data.reset_index(drop=True, inplace=True)

        return self.process_target(case_data=case_data,
                                   case_id=case_id_int,
                                   track_id_list=list(self.target_veh_dict[case_id_str]))

    def save(self, dataframe: pd.DataFrame, file_name: str, dir_=None):
        """
        Save the feature in the data sequence in a single csv files.
        :param dataframe: DataFrame, the dataframe encoded.
        :param file_name: str, the name of csv file
        :param dir_: str, the directory to store the csv file
        :return:
        """
        if not isinstance(dataframe, pd.DataFrame):
            return

        if not dir_:
            dir_ = os.path.join(os.path.split(self.save_dir)[0], "intermediate", self.split + "_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split + "_intermediate", "raw")

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        f_name = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, f_name))

    def process_target(self, case_data: pd.DataFrame, case_id: int, track_id_list: List[int]):
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
        orig = self.traj_dict[track_id][self.obs_horizon - 1].copy().astype(float)
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
        agt_traj_obs = self.traj_dict[track_id][0:self.obs_horizon].copy().astype(float)
        agt_traj_fut = self.traj_dict[track_id][
                       self.obs_horizon: self.obs_horizon+self.pred_horizon].copy().astype(float)

        veh_yaw = obs_track_full[:, [4]].astype(float)
        ctr_line_candts, ctr_path_candts = path_search_rule(track_obs_xy=agt_traj_obs,
                                                            track_obs_yaw=veh_yaw,
                                                            case_data=case_data,
                                                            track_id=track_id,
                                                            hd_map=self.hd_map,
                                                            roundabout=roundabout)

        ctr_line_candts = resample_cl_with_dist(cl_list=ctr_line_candts, dist=0.1)

        original_ctr_line_candts = copy.deepcopy(ctr_line_candts)

        # rotate the center lines and find the reference center line
        agt_traj_obs = np.matmul(rot, (agt_traj_obs - orig.reshape(-1, 2)).T).T
        agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T

        if self.split == "test":
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(cline_list=ctr_line_candts, pred_gt=agt_traj_fut)

        x_min, x_max, y_min, y_max = self.start_obs_lat_range, self.end_obs_lat_range, self.start_obs_lon_range, \
                                     self.end_obs_lon_range

        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []

        # min_feat = None
        # min_has_obs = None
        # min_gt_pred = None
        # min_has_pred = None
        # min_dist = 1e9
        target_feat = None
        target_has_obs = None
        target_gt_pred = None
        target_has_pred = None
        for track_idx in self.traj_dict.keys():
            # if track_idx == track_id:
            #     continue

            traj = copy.deepcopy(self.traj_dict[track_idx])
            step = copy.deepcopy(self.step_dict[track_idx])

            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj-orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), float)
            has_pred = np.zeros(self.pred_horizon, bool)

            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon + self.pred_horizon)
            post_step = step[future_mask]-self.obs_horizon
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
                if step_obs[i] == self.obs_horizon-len(step_obs)+i:
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

            # if step_obs.shape[0] != 10:
            #     print("hello")
            # compare_agt_traj_obs = copy.deepcopy(agt_traj_obs[step_obs])
            # diff_array = traj_obs-compare_agt_traj_obs
            # cur_dist = np.sum(np.hypot(diff_array[:, 0], diff_array[:, 1]))/diff_array.shape[0]

            # if cur_dist < min_dist:
            #     min_dist = cur_dist
            #     min_feat = copy.deepcopy(feat)
            #     min_has_obs = copy.deepcopy(has_obs)
            #     min_has_pred = copy.deepcopy(has_pred)
            #     min_gt_pred = copy.deepcopy(gt_pred)

            # if np.max(feat[:, 0]) < x_min or np.min(feat[:, 0]) > x_max or np.max(feat[:, 1]) < y_min or np.min(feat[:, 1]) > y_max:
            #     continue

            if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max:
                continue
            
            if track_idx == track_id:
                target_feat = copy.deepcopy(feat)
                target_has_obs = copy.deepcopy(has_obs)
                target_gt_pred = copy.deepcopy(gt_pred)
                target_has_pred = copy.deepcopy(has_pred)
            else:
                feats.append(feat)
                has_obss.append(has_obs)
                gt_preds.append(gt_pred)
                has_preds.append(has_pred)

        # if len(feats) == 0:
        #     feats.append(min_feat)
        #     has_obss.append(min_has_obs)
        #     gt_preds.append(min_gt_pred)
        #     has_preds.append(min_has_pred)
        # if len(feats) == 0:
        #     print("hello world")
        feats.insert(0, target_feat)
        has_obss.insert(0, target_has_obs)
        gt_preds.insert(0, target_gt_pred)
        has_preds.insert(0, target_has_pred)
        feats = np.asarray(feats, float)
        has_obss = np.asarray(has_obss, bool)
        gt_preds = np.asarray(gt_preds, float)
        has_preds = np.asarray(has_preds, bool)

        data = dict()

        data["track_id"] = track_id
        data["traj_dict"] = self.traj_dict
        data["full_info_dict"] = self.full_info_dict
        data["step_dict"] = self.step_dict
        data["orig"] = orig
        data["theta"] = theta
        data["rot"] = rot

        data["agt_traj_obs"] = agt_traj_obs
        data["agt_traj_fut"] = agt_traj_fut

        data["feats"] = feats
        data["has_obss"] = has_obss

        data["has_preds"] = has_preds
        data["gt_preds"] = gt_preds

        data["ref_ctr_lines"] = splines
        data["ref_ctr_idx"] = ref_idx

        data["original_ctr_candts"] = original_ctr_line_candts
        data["ctr_candts"] = ctr_line_candts
        data["ctr_path_candts"] = ctr_path_candts

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

            ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, float))
            feats.append(np.asarray(ctrln[1:] - ctrln[:-1], float))

            has_control, turn_direction, is_intersect, speed_limit = self.hd_map.get_lane_info(lane_id=lane_id)
            x = np.zeros((num_segs, 2), float)
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
        graph["speed_limit"] = np.concatenate(speed_limits, 0)

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
            ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in
                               range(len(cline_list))]

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
    save_path = "/home/joe/Dataset/target_data"
    file_list = os.listdir(target_path)

    # scrip command generation
    # str = ""
    # for idx, file in enumerate(file_list):
    #     if idx == 0:
    #         str += f"python interaction_preprocessor_v3.py -m {mode} -f {file}"
    #     else:
    #         str += f" & python interaction_preprocessor_v3.py -m {mode} -f {file}"
    # print(str)

    os.makedirs(os.path.join(save_path, f"{mode}_intermediate", "raw"), exist_ok=True)
    main_func(data_path=os.path.join(data_path, f"{file[:-5]}_{mode}.csv"),
              target_veh_path=os.path.join(target_path, file),
              map_path=os.path.join(map_path, f"{file[:-5]}.osm"),
              scene_name=file[:-5],
              split=mode,
              save_dir=save_path)