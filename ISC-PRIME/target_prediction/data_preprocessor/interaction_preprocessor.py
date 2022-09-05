import copy
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append("/home/joe/Desktop/TRCVTPP/RulePRIMEV2")
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


def get_lane_direction(his_track_full: np.ndarray, hd_map: HDMap):
    obs_traj = his_track_full[:, [0, 1]].astype("float")

    lane_list = hd_map.find_lanelet(pos=obs_traj[-1])

    min_angle = 10e9
    min_yaw = None

    for lane_id in lane_list:
        lane_obj = hd_map.id_lane_dict[lane_id]
        center_line = lane_obj.centerline_array

        cl_ls = LineString(coordinates=center_line)
        point = Point(obs_traj[-1])

        dist = cl_ls.project(point)
        project_p = cl_ls.interpolate(dist)
        if dist != cl_ls.length:
            far_project_p = cl_ls.interpolate(dist+1 if dist+1 < cl_ls.length else cl_ls.length)
            direct_array = np.array([far_project_p.x-project_p.x, far_project_p.y-project_p.y])
        else:
            close_project_p = cl_ls.interpolate(dist-1)
            direct_array = np.array([project_p.x-close_project_p.x, project_p.y-close_project_p.y])

        angle_diff = abs(get_angle(vec_a=direct_array, vec_b=obs_traj[-1]-obs_traj[-2]))

        if angle_diff < min_angle:
            min_yaw = normalize_angle(get_angle(vec_a=np.array([1, 0]), vec_b=direct_array))
            min_angle = angle_diff

    return min_yaw


class InteractionPreprocessor(Preprocessor):
    def __init__(self,
                 root_dir: str,
                 target_veh_direct: str,
                 map_direct: str,
                 split: str = "train",
                 obs_horizon: int = 10,
                 obs_lat_range: int = 30,
                 obs_lon_range: int = 10,
                 pred_horizon: int = 30,
                 normalized: bool = True,
                 save_dir=None):
        super(InteractionPreprocessor, self).__init__(root_dir=root_dir,
                                                      obs_horizon=obs_horizon,
                                                      obs_lat_range=obs_lat_range,
                                                      obs_lon_range=obs_lon_range,
                                                      pred_horizon=pred_horizon)

        self.split = split
        self.normalized = normalized
        self.save_dir = save_dir

        self.target_list = self.get_key_list(target_veh_direct)
        self.map_dict = self.get_map_dict(map_direct)
        self.data_dict = self.get_data_dict(data_dir=root_dir)

    @staticmethod
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

    @staticmethod
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

    def get_data_dict(self, data_dir: str):
        data_dict = {}

        if self.split == "train":
            num_i = 10
        elif self.split == "val" or self.split == "test":
            num_i = 8
        else:
            raise Exception("No this split model: {}".format(self.split))

        data_file_list = os.listdir(data_dir)

        for file in data_file_list:
            scene_name = file[:-num_i]
            data_pandas = DatasetPandas(data_path=os.path.join(data_dir, file))

            data_dict[scene_name] = data_pandas

        return data_dict

    def __getitem__(self, item):
        scene, case_id, track_id = self.target_list[item]

        data_pandas = copy.deepcopy(self.data_dict[scene])
        df = copy.deepcopy(data_pandas.get_case_data(case_id=case_id))
        df.reset_index(drop=True, inplace=True)

        return self.process_and_save(dataframe=df,
                                     seq_id=(scene, case_id, track_id),
                                     dir_=self.save_dir)

    def __len__(self):
        return len(self.target_list)

    def process(self, dataframe: pd.DataFrame, seq_id: Tuple[str, int, int], map_feat: bool = True):
        # print(seq_id)
        # if (seq_id[1], seq_id[2]) == (307, 7):
        #     print("hello")
        data = self.read_interaction_data(df=dataframe, track_id=seq_id[2])
        data = self.get_obj_feats(data=data,
                                  case_data=dataframe,
                                  hd_map=self.map_dict[seq_id[0]],
                                  seq_id=seq_id)

        data["graph"] = self.get_lane_graph(data=data, hd_map=self.map_dict[seq_id[0]])
        data["seq_id"] = f"{seq_id[0]}_{seq_id[1]}_{seq_id[2]}"

        return pd.DataFrame(
            [[data[key] for key in data.keys()]],
            columns=[key for key in data.keys()]
        )

    @staticmethod
    def read_interaction_data(df: pd.DataFrame, track_id: int):
        agt_ts = np.sort(np.unique(df["frame_id"].values))

        mapping = dict()
        for i, ts in enumerate(agt_ts):
            mapping[ts] = i

        trajs = np.concatenate(
            (
                df.x.to_numpy().reshape(-1, 1),
                df.y.to_numpy().reshape(-1, 1)
            ), 1
        )

        full_info = np.concatenate(
            (
                df.x.to_numpy().reshape(-1, 1),
                df.y.to_numpy().reshape(-1, 1),
                df.vx.to_numpy().reshape(-1, 1),
                df.vy.to_numpy().reshape(-1, 1),
                df.psi_rad.to_numpy().reshape(-1, 1),
            ), 1
        )

        steps = [mapping[x] for x in df["frame_id"].values]
        steps = np.asarray(steps, int)

        objs = df.groupby(["track_id"]).groups
        keys = list(objs.keys())

        agt_idx = keys.index(track_id)
        idcs = objs[keys[agt_idx]]

        agt_traj = trajs[idcs]
        agt_full_info = full_info[idcs]
        agt_step = steps[idcs]

        del keys[agt_idx]

        ctx_trajs, ctx_full_info, ctx_steps = [], [], []
        for key in keys:
            idcs = objs[key]
            ctx_trajs.append(trajs[idcs])
            ctx_full_info.append(full_info[idcs])
            ctx_steps.append(steps[idcs])

        data = dict()
        data['trajs'] = [agt_traj] + ctx_trajs
        data['steps'] = [agt_step] + ctx_steps
        data["full_info"] = [agt_full_info] + ctx_full_info
        return data

    def get_obj_feats(self, data: Dict, case_data: pd.DataFrame, hd_map: HDMap, seq_id: Tuple[str, int, int]):
        orig = data["trajs"][0][self.obs_horizon-1].copy().astype(np.float32)
        obs_track_full = data["full_info"][0][:self.obs_horizon].copy().astype(np.float32)

        roundabout = True if "Roundabout" in seq_id[0] else False

        if self.normalized:
            theta = get_lane_direction(his_track_full=obs_track_full, hd_map=hd_map)

            rot = np.asarray([
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)]
            ], np.float32)
        else:
            theta = None

            rot = np.asarray([
                [1.0, 0.0],
                [0.0, 1.0]
            ], np.float32)

        # get the target candidate and candidate gt
        agt_traj_obs = data["trajs"][0][0:self.obs_horizon].copy().astype(np.float32)
        agt_traj_fut = data["trajs"][0][self.obs_horizon: self.obs_horizon+self.pred_horizon].copy().astype(np.float32)

        veh_yaw = obs_track_full[:, [4]].astype(np.float32)

        ctr_line_candts = path_search_rule(track_obs_xy=agt_traj_obs,
                                           track_obs_heading=veh_yaw[:10],
                                           case_data=case_data,
                                           track_id=0,
                                           hd_map=hd_map,
                                           roundabout=roundabout)

        # ======================DEBUG============================
        # if (seq_id[1], seq_id[2]) == (307, 7):
        #     axes = plt.subplot(111)
        #     axes = draw_lanelet_map(hd_map.lanelet_map, axes)
        #     for cl in ctr_line_candts:
        #         axes.plot(cl[:, 0], cl[:, 1], color="gray", linestyle="--")

        #     axes.plot(agt_traj_obs[:, 0], agt_traj_obs[:, 1], color="green")
        #     axes.plot(agt_traj_fut[:, 0], agt_traj_fut[:, 1], color="purple")  
        #     plt.show()
        # ======================DEBUG============================

        # rotate the center lines and find the reference center line
        agt_traj_fut = np.matmul(rot, (agt_traj_fut-orig.reshape(-1, 2)).T).T
        for i, _ in enumerate(ctr_line_candts):
            ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i]-orig.reshape(-1, 2)).T).T

        # candidate sample with center lines, while original TNT sample candidates with boundary line.
        tar_candts = self.lane_candidate_sampling(centerline_list=ctr_line_candts,
                                                  distance=0.5,
                                                  viz=False)

        if self.split == "test":
            tar_candts_gt, tar_offset_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            splines, ref_idx = None, None
        else:
            splines, ref_idx = self.get_ref_centerline(cline_list=ctr_line_candts, pred_gt=agt_traj_fut)
            tar_candts_gt, tar_offset_gt = self.get_candidate_gt(target_candidate=tar_candts, gt_target=agt_traj_fut[-1])
        feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []

        x_min, x_max, y_min, y_max = -self.obs_lat_range, self.obs_lat_range, -self.obs_lon_range, self.obs_lon_range

        for traj, step in zip(data["trajs"], data["steps"]):
            if self.obs_horizon-1 not in step:
                continue

            # normalize and rotate
            traj_nd = np.matmul(rot, (traj-orig.reshape(-1, 2)).T).T

            # collect the future prediction ground truth
            gt_pred = np.zeros((self.pred_horizon, 2), float)
            has_pred = np.zeros(self.pred_horizon, bool)

            future_mask = np.logical_and(step >= self.obs_horizon, step < self.obs_horizon+self.pred_horizon)
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
        data['ref_cetr_idx'] = ref_idx  # the idx of the closest reference centerlines
        return data

    def get_lane_graph(self, data, hd_map: HDMap):
        """
        Get a rectangle area defined by pred range.
        :param data:
        :return:
        """
        x_min, x_max, y_min, y_max = -self.obs_lat_range, self.obs_lat_range, -self.obs_lon_range, self.obs_lon_range
        radius = max(abs(x_min), abs(x_max))+max(abs(y_min), abs(y_max))

        lane_ids = get_lane_id_in_xy_bbox(query_x=data["orig"][0],
                                          query_y=data["orig"][1],
                                          hd_map=hd_map,
                                          query_search_range_manhattan=radius)
        lane_ids = copy.deepcopy(lane_ids)

        # self.plot_graph(hd_map, lane_ids, data["orig"])

        lanes = dict()
        for lane_id in lane_ids:
            lane = hd_map.id_lane_dict[lane_id]
            lane = copy.deepcopy(lane)
            centerline = np.matmul(data["rot"], (lane.centerline_array-data["orig"].reshape(-1, 2)).T).T
            x, y = centerline[:, 0], centerline[:, 1]

            if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                continue
            else:
                """Getting polygons requires original centerline"""
                polygon = get_polygon(lane_obj=lane)
                polygon = copy.deepcopy(polygon)

                lane.centerline = centerline
                lane.polygon = np.matmul(data["rot"], (polygon[:, :2]-data["orig"].reshape(-1, 2)).T).T
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

            has_control, turn_direction, is_intersect, speed_limit = hd_map.get_lane_info(lane_id=lane_id)
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

    def plot_graph(self, hd_map: HDMap, id_list: List[int], veh_pos: np.ndarray):
        axes = plt.subplot(111)

        axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)
        axes = plot_lane_list(id_list=id_list,
                              id_lane_dict=hd_map.id_lane_dict,
                              axes=axes)

        axes.scatter(veh_pos[0], veh_pos[1], color="black", s=25, zorder=20)

        plt.show()

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
                diff = pred_gt-project_p_array
                nl_dist = np.hypot(diff[:, 0], diff[:, 1])
                if nl_dist < min_distance:
                    line_idx = idx

            return ref_centerlines, line_idx


if __name__ == '__main__':
    mode = "test"
    if mode == "test":
        data_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/test_single-agent"
    else:
        data_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"+mode
    target_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/"+mode+"_target_filter/"
    map_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps/"

    preprocessor = InteractionPreprocessor(root_dir=data_path,
                                           target_veh_direct=target_path,
                                           map_direct=map_path,
                                           split=mode,
                                           save_dir="/home/joe/Desktop/TRCVTPP/dataset/rule")

    loader = DataLoader(preprocessor,
                        batch_size=16,
                        num_workers=1,
                        shuffle=False,
                        pin_memory=False,
                        drop_last=False)
    for i, data in enumerate(tqdm(loader)):
        pass
