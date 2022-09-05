import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
import copy
import matplotlib.pyplot as plt

from typing import Tuple


class Preprocessor(Dataset):
    """
    Superclass for all the trajectory data preprocessor.
    Those preprocessors will reformat the data in a single sequence and feed to the system or store them.
    """
    def __init__(self, root_dir: str, obs_horizon: int = 10, obs_lat_range: int = 30, obs_lon_range: int = 15, pred_horizon: int = 30):
        self.root_dir = root_dir

        self.obs_horizon = obs_horizon      # The number of timestamp for observation
        self.obs_lat_range = obs_lat_range          # The observation range
        self.obs_lon_range = obs_lon_range
        self.pred_horizon = pred_horizon      # The number of timestamp for prediction

        self.split = None

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        """
        The total number of sequence in the dataset.
        :return:
        """
        raise NotImplementedError

    def process(self, dataframe: pd.DataFrame, seq_id: Tuple[str, int, int], map_feat: bool = True):
        """
        Select filter the data frame, output filtered data frame
        :param dataframe: DataFrame, the data frame
        :param seq_id: str, the sequence id
        :param map_feat: bool, output map feature or not
        :return: DataFrame [(same as original)]
        """
        raise NotImplementedError

    def extract_feature(self, dataframe: pd.DataFrame, map_feat=True):
        """
        Select and filter the data frame, output filtered by the algorithm
        :param dataframe: DataFrame, the data frame
        :param map_feat: bool, output map feature or not
        :return: DataFrame [(same as original)]
        """
        raise NotImplementedError

    def encode_feature(self, *feats):
        """
        Encode the filtered features to specific format required by the algorithm.
        :param feats: DataFrame, the data frame containing the filtered data
        :return: DataFrame[POLYLINE_FEATURE, GT, TRAJ_ID_TO_MASK, LANE_ID_TO_MASK, TRAJ_LEN, LANE_LEN]
        """
        raise NotImplementedError

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
            dir_ = os.path.join(os.path.split(self.root_dir)[0], "intermediate", self.split+"_intermediate", "raw")
        else:
            dir_ = os.path.join(dir_, self.split+"_intermediate", "raw")

        if not os.path.exists(dir_):
            os.makedirs(dir_)

        f_name = f"features_{file_name}.pkl"
        dataframe.to_pickle(os.path.join(dir_, f_name))

    def process_and_save(self, dataframe: pd.DataFrame, seq_id: Tuple[str, int, int], dir_=None, map_feat=True):
        """
        Save the feature in the date sequence in a single csv files
        :param dataframe: DataFrame, the data frame
        :param seq_id:
        :param dir_: str, the directory to store the csv file
        :param map_feat:
        :return:
        """
        df_processed = self.process(dataframe, seq_id, map_feat)
        self.save(dataframe=df_processed,
                  file_name=f"{seq_id[0]}_{seq_id[1]}_{seq_id[2]}",
                  dir_=dir_)

        return []

    @staticmethod
    def uniform_candidate_sampling(sampling_range, rate=30):
        """
        Uniformly sampling of the target candidate
        :param sampling_range: int, the maximum range of the sampling
        :param rate: the sampling rate (num. of samples)
        :return: rate^2 candidate samples
        """
        x = np.linspace(-sampling_range, sampling_range, rate)
        return np.stack(np.meshgrid(x, x), -1).reshape(-1, 2)

    # implement a candidate sampling with equal distance
    def lane_candidate_sampling(self, centerline_list, distance=0.5, viz=False):
        """
        The input are list of line, each line containing
        :param centerline_list:
        :param distance:
        :param viz:
        :return:
        """
        candidates = []

        for line in centerline_list:
            for i in range(len(line)-1):
                if np.any(np.isnan(line[i])) or np.any(np.isnan(line[i+1])):
                    continue
                [x_diff, y_diff] = line[i+1] - line[i]
                if x_diff == 0.0 and y_diff == 0.0:
                    continue
                candidates.append(line[i])

                # compute displacement along each coordinate
                den = np.hypot(x_diff, y_diff) + np.finfo(float).eps
                d_x = distance * (x_diff / den)
                d_y = distance * (y_diff / den)

                num_c = np.floor(den / distance).astype(np.int)
                pt = copy.deepcopy(line[i])
                for j in range(num_c):
                    pt += np.array([d_x, d_y])
                    candidates.append(copy.deepcopy(pt))
        candidates = np.unique(np.asarray(candidates), axis=0)

        if viz:
            fig = plt.figure(0, figsize=(8, 7))
            fig.clear()
            for centerline_coords in centerline_list:
                plt.plot(centerline_coords[:, 0], centerline_coords[:, 1], color="gray", linestyle="--")
            plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
            plt.xlabel("Map X")
            plt.ylabel("Map Y")
            plt.axis("off")
            plt.title("No. of lane candidates = {}; No. of target candidates = {};".format(len(centerline_list),
                                                                                           len(candidates)))
            plt.show(block=False)

        return candidates

    @staticmethod
    def get_candidate_gt(target_candidate, gt_target):
        """
        Find the target candidate which is closest ot the gt and output the one-hot ground truth.
        :param target_candidate: (N, 2) candidate
        :param gt_target: (1, 2) the coordinate of final target
        :return:
        """
        displacement = gt_target-target_candidate
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

