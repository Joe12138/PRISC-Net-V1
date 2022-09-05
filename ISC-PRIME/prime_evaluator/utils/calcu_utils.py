import logging
from typing import Union, Tuple

import numpy as np
import torch

def calcu_min_ade_fde(preds: np.ndarray, gt: np.ndarray)-> Tuple:

    diff = preds - gt
    disp_error = np.hypot(diff[:, :, 0], diff[:, :, 1])
    min_fde = disp_error[:, -1].min()
    min_ade = np.mean(disp_error, axis=1).min()
    idx = np.argmin(disp_error[:, -1])
    argo_min_ade = np.mean(disp_error[idx])
    return min_fde, min_ade, argo_min_ade

class AverageMeter(object):
    def __init__(self):
        self.reset()       # __init__():reset parameters

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class NearestFinder(object):
    def __init__(self, type, weighted=False):
        if type in ["byMax", "byAvg", "byEndpt", "by2pt", "by3pt"]:
            self.type = type
            self.weighted = weighted
            logging.info(f"Nearest trajectory is specified {type}, weighted={weighted}")
        else:
            assert False, "Unsupported distance metric"

    def get_nearest_with_metric(self, candidates: Union[np.ndarray, torch.tensor],
                                gt: Union[np.ndarray, torch.tensor],
                                tensor: bool = False) -> Union[Tuple[int, float], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the index of the closest trajectory in given candidates
        :param candidates: Pre-generated trajectories. Shape [num_trajs, n_timesteps, state_dim]
        :param gt: Ground truth trajectories of agent. Shape [n_timesteps, state_dim] or [1, n_timesteps, state_dim]
        :param tensor: It's tensor or not (numpy)
        :return: Index of closest traj in the pre-generated trajs
                 (Scaled) Corresponding dist metric
        """
        horizon = gt.shape[0]
        diff = candidates-gt

        if tensor:
            disp_error = torch.sqrt(torch.pow(diff[:, :, 0], 2)+torch.pow(diff[:, :, 1], 2))
        else:
            disp_error = np.hypot(diff[:, :, 0], diff[:, :, 1])

        if self.type == 'byMax':
            dist = disp_error.max(1)
        elif self.type == 'byAvg':
            dist = disp_error.mean(1)
        elif self.type == 'byEndpt':
            dist = disp_error[:, -1]
        elif self.type == 'by2pt':
            if self.weighted:
                dist = (disp_error[:, int(horizon / 2) - 1] + 2 * disp_error[:, -1]) / 3
            else:
                dist = disp_error[:, int(horizon / 2) - 1] + disp_error[:, -1]
        elif self.type == 'by3pt':
            if self.weighted:
                dist = (disp_error[:, int(horizon / 3) - 1] + 2 * disp_error[:,
                                                                  int(horizon * 2 / 3) - 1] + 3 * disp_error[:, -1]) / 6
            else:
                dist = disp_error[:, int(horizon / 3) - 1] + disp_error[:, int(horizon * 2 / 3) - 1] + disp_error[:, -1]

        index = dist.argmin()
        value = dist[index]
        return index, value


class ScoreEvaluator(object):
    """
    Scoring the predicted trajs by some metrics, the resulted scores are used for the Loss of score_classify
    """
    def __init__(self, type, weighted = False):
        if type in ['byMax', 'byAvg', 'byEndpt', 'by2pt', 'by3pt']:
            self.type = type
            self.weighted = weighted
            logging.info(f"Scoring predicted trajectories {type}, weighted={weighted}")
        else:
            assert False, "Unsupported metric for score calculation"

    def scoring_trajs(self, candidates: torch.tensor, gt: torch.tensor, scale = 1.0) -> torch.Tensor:
        ## Here all scores are based on SUM of SQUARED errors (reverted to the oraginal scaling)
        horizon = gt.shape[0]
        diff = (candidates - gt) * scale
        squared_error = torch.pow(diff[:, :, 0], 2) + torch.pow(diff[:, :, 1], 2)
        res = None
        if self.type == 'byMax':
            res = torch.max(squared_error, dim=1)[0]    # [0] - values [1] - indices

        elif self.type == 'byAvg':
            res = torch.mean(squared_error, dim=1)

        elif self.type == 'byEndpt':
            res = squared_error[:, -1]

        elif self.type == 'by2pt':
            if self.weighted:
                res = (squared_error[:, int(horizon/2)-1] + 2 * squared_error[:, -1]) / 3
            else:
                res = (squared_error[:, int(horizon/2)-1] + squared_error[:, -1]) / 2

        elif self.type == 'by3pt':
            if self.weighted:
                res = (squared_error[:, int(horizon/3)-1] + 2 * squared_error[:, int(horizon/3*2)-1] + 3 * squared_error[:, -1]) / 6
            else:
                res = (squared_error[:, int(horizon/3)-1] + squared_error[:, int(horizon/3*2)-1] + squared_error[:, -1]) / 3

        return -1.0 * res



class DispErrorCalculator(object):
    """
    Calculate the displacement error bewteen the predicted trajs and gt by given metrics
    the resulted scores are used for the Loss of pred_regress
    """
    def __init__(self, type):
        if type in ['byMax', 'byAvg', 'byEndpt', 'by2pt']:
            self.type = type
        else:
            assert False, "Unsupported metric for distance calculation"

    def calc_disp_error(self, candidates: torch.tensor, gt: torch.tensor) -> torch.Tensor:
        diff = candidates - gt
        traj_disp_error = torch.norm(diff, dim=-1)
        midpoint = int(gt.shape[0] / 2 - 1)
        if self.type == 'byMax':
            return torch.max(traj_disp_error, dim=1)[0]

        elif self.type == 'byAvg':
            return torch.mean(traj_disp_error, dim=1)

        elif self.type == 'byEndpt':
            return traj_disp_error[:, -1]

        elif self.type == 'by2pt':
            return (traj_disp_error[:, midpoint] + traj_disp_error[:, -1]) / 2

        elif self.type == 'by3pt':
            return (traj_disp_error[:, midpoint] + traj_disp_error[:, -1]) / 2


class PreditionsFilter(object):
    """
    Filter predicted results by end-point (and mid-point) distance constrains
    """
    def __init__(self, endpt_least_dist: float, midpt_least_dist: float):
        self.endpt_dist = endpt_least_dist
        self.midpt_dist = midpt_least_dist

    def update(self, endpt_least_dist: float, midpt_least_dist: float):
        self.endpt_dist = endpt_least_dist
        self.midpt_dist = midpt_least_dist
        # logging.info(f"Filter is updated with least distance: middle={self.midpt_dist:.2f} & end={self.endpt_dist:.2f}")

    def add_predictions_by_least_dist(self, sorted_trajs: np.ndarray, num: int):
        """
        :param sorted_trajs: set of trajs which is already sorted in decreasing priority.
        :return: The selected trajectories which satisfies least distance constrains.
        """
        end_dist, mid_dist = self.endpt_dist, self.midpt_dist
        mid = (sorted_trajs.shape[1] - 1) // 2

        if num >= sorted_trajs.shape[0]:
            result = sorted_trajs
            return result, [id for id in range(sorted_trajs.shape[0])]
        if end_dist<=0.0 and mid_dist<=0.0:
            result = sorted_trajs[:num]
            return result, [id for id in range(num)]

        result, indices = sorted_trajs[[0]], [0]
        while result.shape[0] < num:
            for id, traj in enumerate(sorted_trajs):
                if result.shape[0] == num:
                    return result, indices
                diff = result - traj
                disp = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)
                if all(disp[:, -1]>end_dist) and all(disp[:, mid]>mid_dist):
                    result = np.vstack((result, traj[np.newaxis, :]))
                    indices.append(id)
            end_dist -= 0.20
            mid_dist -= 0.10
        return result, indices


class PreditionsCluster(object):
    """
    Filter the predicted results by clustering with a radius r, output the index of chosen trajs
    """
    def __init__(self, radius: float, cluster_num : int):
        self.r = radius
        self.num = cluster_num
        self.cluster = {}

    def find_max_circle(self, sorted_trajs: np.ndarray, sorted_probs: np.ndarray):
        end_pts = sorted_trajs[:,-1]
        traj_num = len(end_pts)
        dist_table = np.zeros((traj_num, traj_num))

        for i in range(traj_num):
            for j in range(i, traj_num):
                dist_table[i][j] = np.linalg.norm(end_pts[i] - end_pts[j])
        dist_table += dist_table.T - np.diag(dist_table.diagonal())

        # Special cases
        if self.num > len(end_pts):
            return sorted_trajs


        # Create clusters

        while(len(self.cluster) < 6 and self.r > 0):
            trajs_deleted = []
            for cluster_idx in range(self.num):
                max_sum = 0
                for i in range(traj_num):
                    sum = 0
                    idx_list = []

                    if i in trajs_deleted:
                        continue

                    for idx, dist in enumerate(dist_table[i]):
                        if idx in trajs_deleted:
                            continue
                        if dist <= self.r:
                            sum += sorted_probs[idx]
                            idx_list.append(idx)
                    max_sum = max(max_sum, sum)
                    if max_sum == sum:
                        # Save info {cluster_idx: [chosen traj idx, trajs in current cluster]}
                        self.cluster.update({cluster_idx:[i, idx_list]})
                # Dealing with NOT HAVE 6 CLUSTERS
                if cluster_idx in self.cluster.keys():

                    trajs_deleted = trajs_deleted + self.cluster[cluster_idx][1]

            self.r = self.r - 0.25

        if self.r <=0:
            return sorted_trajs[:self.num]

        if len(self.cluster) == 6:
            # Return sorted traj rather than idx
            return [sorted_trajs[self.cluster[key][0]] for key in self.cluster.keys()]
