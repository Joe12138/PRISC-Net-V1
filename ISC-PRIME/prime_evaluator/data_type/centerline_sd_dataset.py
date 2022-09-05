import sys
sys.path.append("/home/joe/Desktop/PredictionWithIRL/Prediction")
import os

import copy
import numpy as np
import pandas as pd
import torch
import logging
from torch.utils.data import DataLoader, Dataset
from typing import Any, List, Tuple, Dict
from prime_evaluator.utils.calcu_utils import NearestFinder
import random
import time

from shapely.geometry import LineString

from prime_evaluator.utils.config import (
    _SCALING_RANGE, _DELTA_YAW_RANGE, _OBS_VARIANCE, _OBS_DROPOUT,
    FEATURE_FILE_PREFIX,
    _CL_SEGMENT_NUM, _SCALING_NUM, _SCALING_D, _SCALING_S

)
from prime_evaluator.utils.visual.viz import (
    viz_centerline_xy_sample
)

from prime_evaluator.utils.parsing import parse_arguments

# RAW_DATA_FORMAT = {
#     "CASE_ID": 0,
#     "VEH_ID": 1,
#     "FRAME_ID": 2,
#     "X": 3,
#     "Y": 4,
#     "VX": 5,
#     "VY": 6,
#     "YAW": 7,
#     "SPEED": 8
# }

RAW_DATA_FORMAT = {
    "CASE_ID": 0,
    "VEH_ID": 1,
    "FRAME_ID": 2,
    "TIMESTAMP": 3,
    "AGENT_TYPE": 4,
    "X": 5,
    "Y": 6,
    "VX": 7,
    "VY": 8,
    "YAW": 9,
    "LENGTH": 10,
    "WIDTH": 11
}


class CenterlineSDDataset(Dataset):
    """Dataset for estimating candidate cl probs"""
    def __init__(self, feature_dir: str, obs_len: int, seq_len: int, key_list: List[Tuple[int, int]],
                 augment: bool, args: Any):
        self.feature_dir = feature_dir
        self.obs_len = obs_len
        self.seq_len = seq_len

        # augment is the core switch of all the sub-augments
        self.scale_augment = augment and (_SCALING_RANGE is not None)
        self.yaw_augment = augment and (_DELTA_YAW_RANGE is not None)
        self.obs_augment = augment and (_OBS_VARIANCE is not None)
        self.obs_dropout = augment and (_OBS_DROPOUT is not None)

        self.args = args
        self.oracle_finder = NearestFinder(type=args.dist_metric, weighted=args.dist_weighted)
        if len(key_list) > 1:
            self.seq_ids = key_list
            self.test = True
        else:
            self.seq_ids = os.listdir(self.feature_dir)
            self.seq_ids.sort()
            self.test = False

        logging.info(
            ' '.join([f"Data Augment-{augment}:",
                      f"scale-{_SCALING_RANGE}" if self.scale_augment else "",
                      f"yaw-{_DELTA_YAW_RANGE}" if self.yaw_augment else "",
                      f"obsVariance-{_OBS_VARIANCE}" if self.obs_augment else "",
                      f"obsDropout-{_OBS_DROPOUT}" if self.obs_dropout else "",
                      ])
        )
        logging.info(f"{len(self.seq_ids)} sequences loaded--------------------")

    def __len__(self):
        return len(self.seq_ids)

    @staticmethod
    def resample_by_length_rule(center_lines: List[np.ndarray]):
        resample_cls_list = []
        for cl in center_lines:
            cl_ls = LineString(cl)
            cl_length = cl_ls.length

            samples = np.linspace(0, cl_length, num=40)

            resample_cl = np.zeros((len(samples), 2), dtype=float)

            for wp_idx, s in enumerate(samples):
                pt = cl_ls.interpolate(s)
                resample_cl[wp_idx][0] = pt.x
                resample_cl[wp_idx][1] = pt.y
            resample_cls_list.append(resample_cl)

        return resample_cls_list

    def __getitem__(self, item):
        """
        transform features to a training for network.
        # AGENT_CLS_FEATURE
        "agent_cls": N_cls x cl_len * 2
        'agent_cls_oracle': List[index of oracle centerline(s)]

        # AGENT_TRAJS_FEATURE
        'agent_track_full' ==>> 'agent_obs_xy': obs_len * 2,
                                'agent_gt_xy': pred_len * 2
        'agent_obs_sds': N_cls x obs_len * 2
        'agent_futs_sd': N_cls x preds_lane * fut_len * 2
        'agent_futs_xy': N_cls x preds_lane * fut_len * 2

        # NBRS_TRAJS_FEATURES (with PADDING indicator in the last column)
        'nbrs_obs_xyz':  N_nbrs * obs_len * 3
        'nbrs_obs_sds': N_cls x N_nbrs * obs_len * 3
        :param item:
        :return:
        """
        seq_id = self.seq_ids[item]
        if self.test:
            path_to_df = os.path.join(self.feature_dir, f"{FEATURE_FILE_PREFIX}{seq_id[0]}_{seq_id[1]}.pkl")
        else:
            path_to_df = os.path.join(self.feature_dir, seq_id)
            # print(path_to_df)
        try:
            df = pd.read_pickle(path_to_df)
        except:
            raise RuntimeError(f"{seq_id} Cannot be read!")

        ## Get from Saved Feature
        agent_cls, agent_cls_oracle = df["AGENT_CLS_FEATURE"].values[0]
        agent_track_full, agent_obs_sds, agent_futs_sd, agent_futs_xy = df["AGENT_TRAJS_FEATURE"].values[0]
        nbrs_obs_xyz, nbrs_obs_sds = df["NBRS_TRAJS_FEATURES"].values[0]

        ## Main feature trans to np.array
        num_cls = len(agent_cls)
        agent_cls = self.resample_by_length_rule(agent_cls)
        agent_cls = np.array([np.pad(cl, ((0, 40 - cl.shape[0]), (0, 0)), 'edge') for cl in
                              agent_cls])  # padding centerline to reach the same number of waypoints
        agent_obs_xy = agent_track_full[:self.obs_len, [RAW_DATA_FORMAT["X"], RAW_DATA_FORMAT["Y"]]].astype("float")
        agent_gt_xy = agent_track_full[self.obs_len:, [RAW_DATA_FORMAT["X"], RAW_DATA_FORMAT["Y"]]].astype(
            "float")  # for test set, ground truth fillted with None/NaN
        agent_futs_xy = [np.array(trajs) for trajs in agent_futs_xy]

        ## Translation wrt obs[-1] (With argment)
        if self.obs_augment: # False
            obs_noise = np.random.multivariate_normal([0, 0], [[_OBS_VARIANCE, 0], [0, _OBS_VARIANCE]], self.obs_len)
            agent_obs_xy = agent_obs_xy + obs_noise

        reference_pt = np.array([agent_obs_xy[-1]])
        agent_cls = np.subtract(agent_cls, reference_pt)
        agent_obs_xy = np.subtract(agent_obs_xy, reference_pt)
        agent_gt_xy = np.subtract(agent_gt_xy, reference_pt)

        ## Scaling (With Augment)----------------------------------------------------------------
        factor = random.uniform(_SCALING_RANGE[0], _SCALING_RANGE[1]) if self.scale_augment else 1.0
        scale = _SCALING_NUM * factor
        agent_obs_xy = agent_obs_xy / scale
        agent_gt_xy = agent_gt_xy / scale
        agent_cls = agent_cls / scale

        ## Rotation (With Augment)----------------------------------------------------------------
        rotation = df["YAW_MAT"].values[0]
        if self.yaw_augment:
            rand_yaw = random.uniform(-1.0, 1.0) * np.deg2rad(_DELTA_YAW_RANGE)
            cos_theta, sin_theta = np.cos(rand_yaw), np.sin(rand_yaw)
            rotation = np.dot(rotation, np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]]))
        agent_obs_xy = agent_obs_xy.dot(rotation)
        agent_gt_xy = agent_gt_xy.dot(rotation)
        agent_cls = agent_cls.dot(rotation)

        ## NBRS Processing ---------------------------------------------# Mark: there may exist some scenes with no nbrs
        nbrs_count = len(nbrs_obs_xyz)
        if nbrs_count > 0:
            nbrs_obs_xyz = np.array(nbrs_obs_xyz)
            nbrs_obs_padding = nbrs_obs_xyz[:, :, -1]
            nbrs_obs_xy = nbrs_obs_xyz[:, :, :2]
            ## Augment
            if self.obs_dropout or self.obs_augment:
                # Drop more observation
                if self.obs_dropout:
                    obs_dropped = np.random.choice([0, 1], p=[1 - _OBS_DROPOUT, _OBS_DROPOUT],
                                                   size=nbrs_obs_padding.shape)
                    nbrs_obs_padding = np.logical_or(nbrs_obs_padding, obs_dropped)
                # Add dist noise
                if self.obs_augment:
                    obs_noise = np.random.multivariate_normal(mean=[0, 0], cov=[[_OBS_VARIANCE, 0], [0, _OBS_VARIANCE]],
                                                              size=(nbrs_count, self.obs_len))
                    nbrs_obs_xy = nbrs_obs_xy + obs_noise
                # Then maintain padding again (make nbrs' padded points the same with its previous, except for the front one)
                for i, padding in enumerate(nbrs_obs_padding):
                    padding_ts = padding.nonzero()[0]
                    for t in padding_ts:
                        near_t = t - 1 if t > 0 else next(j for j, val in enumerate(padding) if val == 0)
                        nbrs_obs_xy[i][t] = nbrs_obs_xy[i][near_t]

            nbrs_obs_xy = np.subtract(nbrs_obs_xy, reference_pt)
            nbrs_obs_xy = nbrs_obs_xy / scale
            nbrs_obs_xy = nbrs_obs_xy.dot(rotation)
            nbrs_obs_sds = np.array(nbrs_obs_sds)[:, :, :, :2]
        else:
            nbrs_obs_padding = np.empty([0, self.obs_len])
            nbrs_obs_xy = np.empty([0, self.obs_len, 2])
            nbrs_obs_sds = np.empty([num_cls, 0, self.obs_len, 2])

        ## Centerlines Processing ----------------------------------------------------------------
        for i in range(num_cls):
            agent_futs_xy[i] = np.dot(np.subtract(agent_futs_xy[i], reference_pt) / scale, rotation)

        fut_oracle_id, _ = self.oracle_finder.get_nearest_with_metric(np.concatenate(agent_futs_xy), agent_gt_xy)

        ## SD coordinator Processing (nbrs is processed before)-----------------------------------
        agent_obs_sds = np.array(agent_obs_sds)
        agent_futs_sd = np.concatenate(agent_futs_sd, axis=0)

        scale_sd = np.array([_SCALING_S * factor, _SCALING_D * factor])
        agent_obs_sds = agent_obs_sds / scale_sd
        nbrs_obs_sds = nbrs_obs_sds / scale_sd
        agent_futs_sd = agent_futs_sd / scale_sd

        ## Finally form a sample -------------------------------------------------
        sample = {'seq_id': seq_id,
                  'city': None,
                  'agent_cls': agent_cls,
                  'agent_cls_oracle': agent_cls_oracle,
                  'agent_obs_xy': agent_obs_xy,
                  'agent_futs_xy': agent_futs_xy,
                  'agent_gt_xy': agent_gt_xy,
                  'fut_oracle_id': fut_oracle_id,
                  'nbrs_obs_xy': nbrs_obs_xy,
                  'nbrs_obs_padding': nbrs_obs_padding,
                  'agent_obs_sds': agent_obs_sds, 'nbrs_obs_sds': nbrs_obs_sds, 'agent_futs_sd': agent_futs_sd,
                  'translation': reference_pt, 'scale': scale, 'rotation': rotation}
        return sample

    def collate_fn(self, batch: List):
        """Custom function for collating a batch of samples
        """
        num_sample = len(batch)
        num_cls = sum([len(sample['agent_cls']) for sample in batch])

        seq_ids = []
        city = []

        scales = []
        translations = []
        rotations = []

        # Related to nbrs
        nbrs_obs_xy = []
        nbrs_obs_padding = []
        nbrs_start_end_idx = np.zeros((num_sample, 2), dtype=int)
        nbrs_accum = 0

        # Related to centerlines
        agent_cls = []
        cls_start_end_idx = np.zeros((num_sample, 2), dtype=int)
        cls_accum = 0
        agent_cls_oracle = []

        # Related to trajectories
        agent_obs_xy = []
        agent_futs_xy = []
        trajs_start_end_idx = np.zeros((num_cls, 2), dtype=int)
        trajs_accum = 0

        # Related to SD representations
        agent_obs_sds = []
        nbrs_obs_sds = []
        agent_futs_sd = []

        # GT
        agent_gt_xy = []
        fut_oracle_idx = []

        for idx, sample in enumerate(batch):
            seq_ids.append(sample['seq_id'])
            city.append(sample['city'])
            scales.append(sample['scale'])
            translations.append(sample['translation'])
            rotations.append(sample['rotation'])

            nbrs_obs_xy.append(sample['nbrs_obs_xy'])
            nbrs_obs_padding.append(sample['nbrs_obs_padding'])
            nbrs_start_end_idx[idx][0] = nbrs_accum
            nbrs_accum += sample['nbrs_obs_xy'].shape[0]      # First & last nbrs
            nbrs_start_end_idx[idx][1] = nbrs_accum

            agent_cls.append(sample['agent_cls'])
            agent_cls_oracle.append(sample['agent_cls_oracle'])
            cls_start_end_idx[idx][0] = cls_accum             # First & last centerlines
            cls_start_end_idx[idx][1] = cls_accum + sample['agent_cls'].shape[0]

            agent_obs_xy.append(sample['agent_obs_xy'])
            agent_gt_xy.append(sample['agent_gt_xy'])
            fut_oracle_idx.append(sample['fut_oracle_id'])
            for trajs in sample['agent_futs_xy']:
                agent_futs_xy.append(trajs)
                trajs_start_end_idx[cls_accum][0] = trajs_accum
                trajs_accum += trajs.shape[0]                 # First & last predictions
                trajs_start_end_idx[cls_accum][1] = trajs_accum
                cls_accum += 1

            agent_obs_sds.append(sample['agent_obs_sds'])
            nbrs_obs_sds.append(sample['nbrs_obs_sds'])
            agent_futs_sd.append(sample['agent_futs_sd'])

        # Merge all samples to a collated batch
        scales = np.array(scales)  # batch
        translations = np.concatenate(translations, axis=0)  # batch x 2
        rotations = np.stack(rotations, axis=0)  # batch x 2 x 2

        agent_cls = np.concatenate(agent_cls, axis=0)
        if agent_cls_oracle[0]==None:
            agent_cls_oracle = np.array([False] * cls_accum)     # In test mode
        else:
            agent_cls_oracle = np.array(
                [ 1 if id in oracle_idx else 0
                  for (id_s, id_e), oracle_idx in zip(cls_start_end_idx, agent_cls_oracle)
                  for id in range(id_e - id_s)]
            )   # Turn to boolean to keep it with the same length with agent_cls
        nbrs_obs_xy = np.concatenate(nbrs_obs_xy, axis=0)
        nbrs_obs_padding = np.concatenate(nbrs_obs_padding, axis=0)
        agent_obs_xy = np.stack(agent_obs_xy, axis=0)
        agent_futs_xy = np.concatenate(agent_futs_xy, axis=0)
        agent_gt_xy = np.stack(agent_gt_xy, axis=0)
        fut_oracle_idx = np.array(fut_oracle_idx)

        agent_obs_sds = np.concatenate(agent_obs_sds, axis=0)
        agent_futs_sd = np.concatenate(agent_futs_sd, axis=0)
        nbrs_obs_sds = np.concatenate([sds.reshape([-1, self.obs_len, 2]) for sds in nbrs_obs_sds], axis=0)

        # print('nbrs_start_end_idx:\n', nbrs_start_end_idx)
        # print('cls_start_end_idx:\n', cls_start_end_idx)
        # print('trajs_start_end_idx:\n', trajs_start_end_idx)
        sample_valid = True
        sample_batched = {
            # If torch.is_tensor(key_name) then will be converted to cuda()
            'seq_ids': seq_ids,
            'city': city,
            'scales': torch.from_numpy(scales).float(),
            'translations': torch.from_numpy(translations).float(),
            'rotations': torch.from_numpy(rotations).float(),

            'cls_start_end_idx': cls_start_end_idx,
            'nbrs_start_end_idx': nbrs_start_end_idx,
            'trajs_start_end_idx': trajs_start_end_idx,

            'agent_cls': torch.from_numpy(agent_cls).float(),
            'agent_cls_oracle': torch.from_numpy(agent_cls_oracle).float(),

            'nbrs_obs_xy': torch.from_numpy(nbrs_obs_xy).float(),
            'nbrs_obs_padding': torch.from_numpy(nbrs_obs_padding).float(),
            'agent_obs_xy': torch.from_numpy(agent_obs_xy).float(),
            'agent_futs_xy': torch.from_numpy(agent_futs_xy).float(),
            'agent_gt_xy': torch.from_numpy(agent_gt_xy).float(),
            'fut_oracle_idx': torch.from_numpy(fut_oracle_idx).int(),

            'agent_obs_sds':torch.from_numpy(agent_obs_sds).float(),
            'agent_futs_sd': torch.from_numpy(agent_futs_sd).float(),
            'nbrs_obs_sds': torch.from_numpy(nbrs_obs_sds).float(),
        }
        return sample_valid, sample_batched


    def get_index_by_seq_id(self, seq_id: Tuple[int, int]):
        idx = self.seq_ids.index(seq_id)
        return idx


    def get_ground_truth(self, sample_batched):
        """
        Here the batched sample (with np.ndarray format) have not been transformed
        """
        gt = {}
        city_names = {}
        seq_ids = sample_batched['seq_ids']
        for i, seq_id in enumerate(seq_ids):
            gt[seq_id] = sample_batched['agent_gt_xy'][i].dot(sample_batched['rotations'][i].transpose()) \
                         * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
            city_names[seq_id] = sample_batched['city'][i]
        return gt, city_names


def trans_back_sample_batched(sample_batched: Dict[str, Any])-> Dict[str, Any]:
    """
    Transform the batched sample to its original coordinate (Manipulate on the copied data)
    Only called (in visualization) before everything related to calculation have been completed.
    """
    sample_batched = copy.deepcopy(sample_batched)
    cls_idx = sample_batched['cls_start_end_idx']
    trajs_idx = sample_batched['trajs_start_end_idx']
    nbrs_idx = sample_batched['nbrs_start_end_idx']

    for i, start_end in enumerate(cls_idx):
        # Agent obs
        sample_batched['agent_obs_xy'][i, :, :] = \
            sample_batched['agent_obs_xy'][i, :, :].dot(sample_batched['rotations'][i].transpose()) \
            * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
        # Agent ground truth
        sample_batched['agent_gt_xy'][i, :, :] = \
            sample_batched['agent_gt_xy'][i, :, :].dot(sample_batched['rotations'][i].transpose()) \
            * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
        # Nbr obs
        for nbr_id in range(nbrs_idx[i][0], nbrs_idx[i][1]):
            sample_batched['nbrs_obs_xy'][nbr_id, :, :] = \
                sample_batched['nbrs_obs_xy'][nbr_id, :, :].dot(sample_batched['rotations'][i].transpose()) \
                * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
        # Centerlines
        for cl_id in range(start_end[0], start_end[1]):
            sample_batched['agent_cls'][cl_id, :, :] = \
                sample_batched['agent_cls'][cl_id, :, :].dot(sample_batched['rotations'][i].transpose()) \
                * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
        # Future predictions
        for fut_id in range(trajs_idx[start_end[0]][0], trajs_idx[start_end[1]-1][1]):
            sample_batched['agent_futs_xy'][fut_id, :, :] = \
                sample_batched['agent_futs_xy'][fut_id, :, :].dot(sample_batched['rotations'][i].transpose()) \
                * sample_batched['scales'][i] + sample_batched['translations'][[i], :]
    return sample_batched


#########################################################
#################### Checking Funcs #####################
#########################################################
def dataset_self_check(args):
    key_list = [(1, 4), (1, 5), (1, 8), (1, 9)]
    cl_xy_dataset = CenterlineSDDataset(feature_dir=args.dataset, obs_len=10, seq_len=40, augment=True, args=args,
                                        key_list=key_list)
    cl_xy_dataloader = DataLoader(cl_xy_dataset,
                               batch_size=args.batch_size,
                               shuffle=True,
                               num_workers=args.num_workers,
                               collate_fn=cl_xy_dataset.collate_fn)
    start = time.time()
    print("Testing all the data")
    for i, (sample_valid, sample_batched) in enumerate(cl_xy_dataloader):
        for key in sample_batched:
            if torch.is_tensor(sample_batched[key]):
                sample_batched[key] = sample_batched[key].cuda()
        print(f"{sample_batched['seq_ids'][0]}")

    print("Testing DONE: {:.2f}s".format(time.time()-start))


def dataset_sample_plot(args, save_in=None, num=None):
    key_list = [(1, 4), (1, 5), (1, 8), (1, 9)]
    cl_xy_dataset = CenterlineSDDataset(feature_dir=args.dataset, obs_len=10, seq_len=40, augment=True, args=args,
                                        key_list=key_list)
    vis_num = num or len(cl_xy_dataset)
    if save_in:
        os.makedirs(save_in, exist_ok=True)
        print(f"Save {vis_num} sample images in {save_in}")
    for i in range(vis_num):
        viz_centerline_xy_sample(cl_xy_dataset[i], save_loc=save_in)


if __name__ == "__main__":
    args = parse_arguments()
    start = time.time()
    key_list = [(1, 4), (1, 5), (1, 8), (1, 9)]
    cl_xy_dataset = CenterlineSDDataset(feature_dir=args.dataset, obs_len=10, seq_len=40, augment=True, args=args,
                                        key_list=key_list)
    print("Dataset init: {:.2f}s".format(time.time() - start))

    dataset_self_check(args)
    dataset_sample_plot(args=args, save_in="/home/joe/Desktop/Test")