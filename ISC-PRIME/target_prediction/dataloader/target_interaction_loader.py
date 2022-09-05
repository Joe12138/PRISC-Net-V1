import math
import sys
sys.path.append("/home/joelan/Desktop/TRCVTPP/RulePRIMEV2")

import gc
import os
from copy import copy
import re

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, InMemoryDataset, DataLoader

from target_prediction.dataloader.graph_data import GraphData
from target_prediction.dataloader.interaction_loader import get_fc_edge_index


class TargetInteractionInMem(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(TargetInteractionInMem, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        gc.collect()

    @property
    def raw_file_names(self):
        # change the loader range
        file_list = [file for file in os.listdir(self.raw_dir) if "features" in file and file.endswith("pkl")]
        return file_list

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes
        turn: two binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        polyline_id: the polyline id of this node belonging to;
        """
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        traj_feats = data_seq["feats"].values[0]
        traj_has_obss = data_seq["has_obss"].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0

        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))

            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn,
                                                 is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turns = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue  # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)

    def process(self):
        """
        Transform the raw data and store in GraphData
        :return:
        """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []

        for raw_path in tqdm(self.raw_paths, desc="Loading Raw Data..."):
            """
            data["orig"]: the original coordinate of the last observation position
            data["theta"]: lane_direction
            data["rot"]: rotation matrix

            data["feats"]: observation trajectory
            data["has_obss"]:

            data['has_preds']:
            data['gt_preds']:
            data['tar_candts']:
            data['gt_candts']:
            data['gt_tar_offset']:

            data['ref_ctr_lines']:
            data['ref_cetr_idx']: the idx of the closest reference center lines


            data['trajs']
            data['steps']
            data["full_info"]

            data["graph"]:
                graph['ctrs'] = np.concatenate(ctrs, 0)
                graph['num_nodes'] = num_nodes
                graph['feats'] = np.concatenate(feats, 0)
                graph['turn'] = np.concatenate(turn, 0)
                graph['control'] = np.concatenate(control, 0)
                graph['intersect'] = np.concatenate(intersect, 0)
                graph['lane_idcs'] = lane_idcs
            data["seq_id"]
            """
            raw_data = pd.read_pickle(raw_path)

            # statistics
            traj_num = raw_data["feats"].values[0].shape[0]
            traj_lens.append(traj_num)

            lane_num = raw_data["graph"].values[0]["lane_idcs"].max() + 1
            valid_lens.append(traj_num + lane_num)

            candidate_num = raw_data["tar_candts"].values[0].shape[0]
            candidate_lens.append(candidate_num)

        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)
        print("/n[Interaction]: The maximum of valid length is {}.".format(num_valid_len_max))
        print("[Interaction]: The maximum of no. of candidates is {}.".format(num_candidate_max))

        # pad vector to the largest polyline id and extend cluster, save the Data to disk
        data_list = []
        for ind, raw_path in enumerate(tqdm(self.raw_paths, desc="Transforming the data to GraphData...")):
            raw_data = pd.read_pickle(raw_path)

            # input data
            x, cluster, edge_index, identifier = self._get_x(raw_data)
            y = self._get_y(raw_data)

            yaw_candts, yaw_array, vx_candts, vx_array, vy_candts, vy_array = self.get_yaw_velocity_candidate(raw_data)

            yaw_candts_gt, yaw_offset_gt = self.get_candidate_gt(yaw_candts, yaw_array[-1])
            vx_candts_gt, vx_offset_gt = self.get_candidate_gt(vx_candts, vx_array[-1])
            vy_candts_gt, vy_offset_gt = self.get_candidate_gt(vy_candts, vy_array[-1])

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
                candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
                candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
                offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
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
                vx_offset_gt=torch.from_numpy(vx_offset_gt).float()
            )
            data_list.append(graph_input)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def get_yaw_velocity_candidate(self, raw_data):
        full_info = raw_data["full_info"][0][0].copy().astype(float)

        vx_array = full_info[:, [2]]
        vy_array = full_info[:, [3]]
        yaw_array = full_info[:, [4]]

        cur_yaw = yaw_array[9][0]
        cur_vx = vx_array[9][0]
        cur_vy = vy_array[9][0]

        yaw_candts = np.linspace(cur_yaw-math.pi/2, cur_yaw+math.pi/2, 21)
        vx_candts = np.linspace(cur_vx-5, cur_vx+5, 21)
        vy_candts = np.linspace(cur_vy-5, cur_vy+5, 21)

        return np.asarray(yaw_candts).reshape((-1, 1)), yaw_array, np.asarray(vx_candts).reshape((-1, 1)), vx_array, \
               np.asarray(vy_candts).reshape((-1, 1)), vy_array

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        displacement = gt_target-target_candidate
        gt_index = np.argmin(abs(displacement))

        one_hot = np.zeros((target_candidate.shape[0], 1))
        one_hot[gt_index] = 1

        offset = gt_target-target_candidate[gt_index]

        return one_hot, offset

    @staticmethod
    def get_candidate_gt_v2(target_candidate, gt_target):
        displacement = gt_target - target_candidate
        gt_index = np.argmin(abs(displacement))

        one_hot = np.zeros((target_candidate.shape[0], 1))
        one_hot[gt_index] = 1

        offset = gt_target - target_candidate

        return one_hot, offset

    def get(self, idx):
        data = super(TargetInteractionInMem, self).get(idx).clone()

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

    data_path = "/home/joe/Dataset/rule_test/DR_CHN_Roundabout_LN/val_intermediate/"

    in_mem_data = TargetInteractionInMem(data_path)

    batch_iter = DataLoader(in_mem_data, batch_size=2, num_workers=0, shuffle=False, pin_memory=True)

    for i, data in enumerate(tqdm(batch_iter, total=len(batch_iter), bar_format="{l_bar}{r_bar}")):
        pass