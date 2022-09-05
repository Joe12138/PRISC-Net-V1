import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import os
import numpy as np
import pandas as pd
import torch
import gc
import math

import copy
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Union, List, Tuple
from shapely.geometry import LineString, Point
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

from target_prediction.dataloader.interaction_loader import get_fc_edge_index
from target_prediction.dataloader.graph_data import GraphData


class InteractionInMem(InMemoryDataset):
    def __init__(self, root, split: str = "train", transform=None, pre_transform=None):
        self.split = split
        super(InteractionInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # self.data, self.slices = torch.load("/home/joe/Dataset/test_folder/val_intermediate/processed/data")
        gc.collect()

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        # change the load range
        file_list = [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith("pkl")]
        return file_list

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ["data.pt"]

    def download(self):
        pass

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, (speed_limit), polyline_id];
        xs, ys: the control point of the vector,
                for trajectory, it's start point,
                for lane segment, it's the ceneter point
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0
        traffic_control: feature for lanes        turn: two binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        (speed_limit): indicate the speed limit of this lane
        polyline_id: the polyline id of this node belonging to;
        """
        # If add speed limit
        # feats = np.empty((0, 11))
        # axes = plt.subplot(111)
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        traj_feats = data_seq["feats"].values[0]
        traj_cnt = 0

        traj_has_obss = data_seq["has_obss"].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2]-feat[has_obs][:-1, :2]

            # for ips in range(xy_s.shape[0]):
            #     axes.annotate("", xy=xy_s[ips], xytext=xy_s[ips]+vec[ips], arrowprops=dict(arrowstyle="->", color="purple"))
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))

            # If add speed limit
            # speed_limit = np.zeros((len(xy_s), 1))
            polyline_id = np.ones((len(xy_s), 1))*traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn,
                                                 is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq["graph"].values[0]
        ctrs = graph["ctrs"]
        vec = graph["feats"]

        # for ips in range(ctrs.shape[0]):
        #     axes.annotate("", xy=ctrs[ips], xytext=ctrs[ips] + vec[ips], arrowprops=dict(arrowstyle="->"))
        
        # plt.xlim((-150, 150))
        # plt.ylim((-150, 150))

        traffic_ctrl = graph["control"].reshape(-1, 1)
        is_turn = graph["turn"]
        is_intersect = graph["intersect"].reshape(-1, 1)
        lane_idcs = graph["lane_idcs"].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turn, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge index
        cluster = copy.copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])

            if len(indices) <= 1:
                continue # skip if only 1 node

            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])

        # plt.show()
        # plt.cla()
        # plt.clf()
        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obj = data_seq["feats"].values[0][0]
        traj_fut = data_seq["gt_preds"].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :]-traj_obj[-1, :2], traj_fut[1:, :]-traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(float)

    @staticmethod
    def lane_candidate_sampling_rule(centerline_list: List[np.ndarray], cur_pos: np.ndarray, distance: float = 0.5):
        """
        The input are list of line, each line containing
        """
        candidates = []
        for line in centerline_list:
            cl_ls = LineString(coordinates=line)
            cur_point = Point(cur_pos)
            cur_dist = cl_ls.project(cur_point)

            candts_dist = cur_dist + distance
            if candts_dist > cl_ls.length:
                # candts_point = cl_ls.interpolate(cl_ls.length)
                candidates.append((line[-1][0], line[-1][0]))
            else:
                while candts_dist <= cl_ls.length:
                    candts_point = cl_ls.interpolate(candts_dist)
                    candidates.append((candts_point.x, candts_point.y))

                    candts_dist += distance
        return np.asarray(candidates)

    @staticmethod
    def lane_candidate_sampling(centerline_list: List[np.ndarray], distance: float = 0.5):
        candidates = []

        for line in centerline_list:
            for i in range(len(line) - 1):
                if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i + 1])):
                    continue
                [x_diff, y_diff] = line[i + 1] - line[i]
                if x_diff == 0.0 and y_diff == 0.0:
                    continue
                candidates.append(line[i])

                # compute displacement along each coordinate
                den = np.hypot(x_diff, y_diff) + np.finfo(float).eps
                d_x = distance * (x_diff / den)
                d_y = distance * (y_diff / den)

                num_c = np.floor(den / distance).astype(int)
                pt = copy.deepcopy(line[i])
                for j in range(num_c):
                    pt += np.array([d_x, d_y])
                    candidates.append(copy.deepcopy(pt))
        candidates = np.unique(np.asarray(candidates), axis=0)

        return candidates

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        Find the target candidate which is closest ot the gt and output the one-hot ground truth.
        :param target_candidate: (N, 2) candidate
        :param gt_target: (1, 2) the coordinate of final target
        :return:
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate[gt_index]
        return onehot, offset_xy

    @staticmethod
    def get_candidate_gt_v2(target_candidate, gt_target):
        """
        Find the target candidate which is closest ot the gt and output the one-hot ground truth.
        :param target_candidate: (N, 2) candidate
        :param gt_target: (1, 2) the coordinate of final target
        :return:
        """
        displacement = gt_target - target_candidate
        gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

        onehot = np.zeros((target_candidate.shape[0], 1))
        onehot[gt_index] = 1

        offset_xy = gt_target - target_candidate
        return onehot, offset_xy

    @staticmethod
    def get_yaw_vel_candidate_gt(target_candidate, gt_target):
        displacement = gt_target-target_candidate
        gt_index = np.argmin(abs(displacement))

        one_hot = np.zeros((target_candidate.shape[0], 1))
        one_hot[gt_index] = 1

        offset = gt_target-target_candidate[gt_index]

        return one_hot, offset

    @staticmethod
    def get_yaw_vel_candidate_gt_v2(target_candidate, gt_target):
        displacement = gt_target - target_candidate
        gt_index = np.argmin(abs(displacement))

        one_hot = np.zeros((target_candidate.shape[0], 1))
        one_hot[gt_index] = 1

        offset = gt_target - target_candidate

        return one_hot, offset

    def get_yaw_velocity_candidate(self, raw_data):
        track_id = raw_data["track_id"].values[0].astype(int)
        full_info = raw_data["full_info_dict"].values[0][track_id]

        vx_array = full_info[:, [2]]
        vy_array = full_info[:, [3]]
        yaw_array = full_info[:, [4]]

        cur_yaw = yaw_array[9][0]
        cur_vx = vx_array[9][0]
        cur_vy = vy_array[9][0]

        yaw_candts = np.linspace(cur_yaw - math.pi / 2, cur_yaw + math.pi / 2, 21)
        vx_candts = np.linspace(cur_vx - 5, cur_vx + 5, 21)
        vy_candts = np.linspace(cur_vy - 5, cur_vy + 5, 21)

        return np.asarray(yaw_candts).reshape((-1, 1)), yaw_array, np.asarray(vx_candts).reshape((-1, 1)), vx_array, \
               np.asarray(vy_candts).reshape((-1, 1)), vy_array

    def process(self):
        """
        Transform the raw data and store in GraphData
        """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []

        candidate_dict = {}
        candidate_gt_dict = {}
        offset_gt_dict = {}

        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            """
            data["track_id"]: the track id of the data sequence
            data["traj_dict"]: the all trajectories of this case data
            data["full_info_dict"]: the full vehicle state information of the target vehicle [x, y, vx, vy, psi_rad]
            data["step_dict"]: the step dict of all vehicle in this case data
            data["orig"]: the original coordinate of target vehicle at 10 timestamp
            data["theta"]: the direction of the lane
            data["rot"]: the rotation matrix
            data["feats"]: 
            data["has_obss"]:
            data["has_preds"]:
            data["gt_preds"]:
            data["ref_ctr_lines"]:
            data["ref_ctr_idx"]:
            data["original_ctr_candts"]
            data["ctr_candts"]:
            data["ctr_path_candts"]:
            
            data["graph"]:
                        ["ctrs"]:
                        ["num_nodes"]:
                        ["feats"]:
                        ["turn"]:
                        ["control"]
                        ["intersect"]
                        ["lane_idcs"]
                        ["speed_limit"]
            """
            raw_data = pd.read_pickle(raw_path)

            # statistics
            traj_num = raw_data["feats"].values[0].shape[0]
            traj_lens.append(traj_num)

            lane_num = raw_data["graph"].values[0]["lane_idcs"].max()+1
            valid_lens.append(traj_num+lane_num)

            ctr_line_candts = raw_data["ctr_candts"].values[0]
            tar_candts = self.lane_candidate_sampling(centerline_list=ctr_line_candts,
                                                      distance=0.5)
            # rule target_sample
            # agt_traj_obs = raw_data["agt_traj_obs"].values[0]
            # tar_candts = self.lane_candidate_sampling_rule(centerline_list=ctr_line_candts,
            #                                                cur_pos=agt_traj_obs[-1],
            #                                                distance=0.5)

            if self.split == "test":
                tar_candts_gt, tar_offset_gt = np.zeros((tar_candts.shape[0], 1)), np.zeros((1, 2))
            else:
                agt_traj_fut = raw_data["agt_traj_fut"].values[0]
                tar_candts_gt, tar_offset_gt = self.get_candidate_gt(target_candidate=tar_candts,
                                                                     gt_target=agt_traj_fut[-1])

            candidate_dict[raw_path] = tar_candts
            candidate_gt_dict[raw_path] = tar_candts_gt
            offset_gt_dict[raw_path] = tar_offset_gt

            candidate_num = tar_candts.shape[0]
            candidate_lens.append(candidate_num)

        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)

        print("/n[Interaction]: The maximum of valid length is {}.".format(num_valid_len_max))
        print("[Interaction]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            raw_data = pd.read_pickle(raw_path)

            # input data
            x, cluster, edge_index, identifier = self._get_x(raw_data)
            y = self._get_y(raw_data)

            yaw_candts, yaw_array, vx_candts, vx_array, vy_candts, vy_array = self.get_yaw_velocity_candidate(raw_data)

            yaw_candts_gt, yaw_offset_gt = self.get_yaw_vel_candidate_gt(yaw_candts, yaw_array[-1])
            vx_candts_gt, vx_offset_gt = self.get_yaw_vel_candidate_gt(vx_candts, vx_array[-1])
            vy_candts_gt, vy_offset_gt = self.get_yaw_vel_candidate_gt(vy_candts, vy_array[-1])

            graph_input = GraphData(
                x=torch.from_numpy(x).float(),
                y=torch.from_numpy(y).float(),
                cluster=torch.from_numpy(cluster).short(),
                edge_index=torch.from_numpy(edge_index).long(),
                identifier=torch.from_numpy(identifier).float(),  # the identify embedding of global graph completion

                traj_len=torch.tensor([traj_lens[ind]]).int(),  # number of traj polyline
                valid_len=torch.tensor([valid_lens[ind]]).int(),  # number of valid polyline
                time_step_len=torch.tensor([num_valid_len_max]).int(),  # the maximum of no. of polyline

                candidate_len_max=torch.tensor([num_candidate_max]).int(),
                candidate_mask=[],
                candidate=torch.from_numpy(candidate_dict[raw_path]).float(),
                candidate_gt=torch.from_numpy(candidate_gt_dict[raw_path]).bool(),
                offset_gt=torch.from_numpy(offset_gt_dict[raw_path]).float(),
                target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),

                orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
                rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
                seq_id=raw_data['seq_id'],

                yaw_candts=torch.from_numpy(yaw_candts).float(),
                vx_candts=torch.from_numpy(vx_candts).float(),
                vy_candts=torch.from_numpy(vy_candts).float(),

                yaw_array=torch.from_numpy(yaw_array).float(),
                vx_array=torch.from_numpy(vx_array).float(),
                vy_array=torch.from_numpy(vy_array).float(),

                yaw_candts_gt=torch.from_numpy(yaw_candts_gt).float(),
                vx_candts_gt=torch.from_numpy(vx_candts_gt).float(),
                vy_candts_gt=torch.from_numpy(vy_candts_gt).float(),

                yaw_offset_gt=torch.from_numpy(yaw_offset_gt).float(),
                vy_offset_gt=torch.from_numpy(vy_offset_gt).float(),
                vx_offset_gt=torch.from_numpy(vx_offset_gt).float(),

                agt_obs_xy=torch.from_numpy(raw_data["agt_traj_obs"].values[0]).float()
            )
            data_list.append(graph_input)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get(self, idx):
        data = super(InteractionInMem, self).get(idx).clone()

        feature_len = data.x.shape[1]
        index_to_pad = data.time_step_len[0].item()
        valid_len = data.valid_len[0].item()

        # pad feature with zero nodes
        data.x = torch.cat([data.x, torch.zeros((index_to_pad - valid_len, feature_len), dtype=data.x.dtype)])
        data.cluster = torch.cat([data.cluster, torch.arange(valid_len, index_to_pad, dtype=data.cluster.dtype)])
        data.identifier = torch.cat(
            [data.identifier, torch.zeros((index_to_pad - valid_len, 2), dtype=data.identifier.dtype)])

        # pad candidate and candidate_gt
        num_cand_max = data.candidate_len_max[0].item()
        data.candidate_mask = torch.cat([torch.ones((len(data.candidate), 1)),
                                         torch.zeros((num_cand_max - len(data.candidate), 1))])
        data.candidate = torch.cat([data.candidate, torch.zeros((num_cand_max - len(data.candidate), 2))])
        data.candidate_gt = torch.cat([data.candidate_gt,
                                       torch.zeros((num_cand_max - len(data.candidate_gt), 1),
                                                   dtype=data.candidate_gt.dtype)])

        return data


if __name__ == '__main__':
    data_path = "/home/joe/ServerBackup/final_version_rule/train_intermediate/"

    in_mem_data = InteractionInMem(data_path)

    batch_iter = DataLoader(in_mem_data, batch_size=512, num_workers=2, shuffle=False, pin_memory=True)

    for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
        pass