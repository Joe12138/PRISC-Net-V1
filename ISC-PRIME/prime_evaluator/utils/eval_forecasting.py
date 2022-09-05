# <Copyright 2019, Argo AI, LLC. Released under the MIT license.>

"""This module evaluates the forecasted trajectories against the ground truth."""

import math
import numpy as np
import pickle as pkl
from typing import Dict, List, Optional, Tuple, Union

from prime_evaluator.utils.config import (
    _MAX_GUESSES_NUM, _MISSING_THRESHOLD,
)

LOW_PROB_THRESHOLD_FOR_METRICS = 0.05


def get_accuracy_statistic(dataset,
                           pred_horizen: int,
                           pred_trajs:dict,
                           sample_batched: dict,
                           fixed_guesses_num: Union[None, int] = None,
                           postfix: str = None):
    """
    :param fixed_guesses_num:   If set, only perform accuracy calculation under a fixed number of guesses
    :param postfix:             If set, add the postfix to all the keys
    :return:                    The resulted accuracy statistic
    """
    metrics = {}
    miss_threshold = _MISSING_THRESHOLD
    if isinstance(pred_trajs, dict):
        gt, _ = dataset.get_ground_truth(sample_batched)
        guesses_choices = [1, _MAX_GUESSES_NUM] \
            if fixed_guesses_num is None else [fixed_guesses_num]
        for max_guesses in guesses_choices:
            res = get_displacement_errors_and_miss_rate(pred_trajs,
                                                        gt,
                                                        max_guesses=max_guesses,
                                                        horizon=pred_horizen,
                                                        miss_threshold=miss_threshold)
            if postfix is None:
                metrics.update(res)
            else:
                for k, v in res.items():
                    metrics[k + postfix] = v
    return metrics


def get_fde_statistic(dataset,
                      pred_horizen: int,
                      pred_trajs:dict,
                      sample_batched: dict):
    gt_trajs, _ = dataset.get_ground_truth(sample_batched)
    fde_stat = []
    for k, gt_traj in gt_trajs.items():
        fde = min([get_fde(pred_traj[:pred_horizen], gt_traj[:pred_horizen]) for pred_traj in pred_trajs[k]])
        fde_stat.append(fde)
    return fde_stat


def get_mde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Maximum Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        mde: Maximum Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    mde = float(
        max(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
    )
    return mde


def get_ade(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Average Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        ade: Average Displacement Error

    """
    pred_len = forecasted_trajectory.shape[0]
    ade = float(
        sum(
            math.sqrt(
                (forecasted_trajectory[i, 0] - gt_trajectory[i, 0]) ** 2
                + (forecasted_trajectory[i, 1] - gt_trajectory[i, 1]) ** 2
            )
            for i in range(pred_len)
        )
        / pred_len
    )
    return ade


def get_fde(forecasted_trajectory: np.ndarray, gt_trajectory: np.ndarray) -> float:
    """Compute Final Displacement Error.

    Args:
        forecasted_trajectory: Predicted trajectory with shape (pred_len x 2)
        gt_trajectory: Ground truth trajectory with shape (pred_len x 2)

    Returns:
        fde: Final Displacement Error

    """
    fde = math.sqrt(
        (forecasted_trajectory[-1, 0] - gt_trajectory[-1, 0]) ** 2
        + (forecasted_trajectory[-1, 1] - gt_trajectory[-1, 1]) ** 2
    )
    return fde


def get_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade = [], []
    min_fde, prob_min_fde = [], []
    min_mde = []
    min_ade_for_min_fde, prob_min_ade_for_min_fde = [], []
    n_misses_by_fde, prob_n_misses_by_fde = [], []
    n_misses_by_mde = []

    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        curr_min_mde = float("inf")
        curr_min_ade_for_min_fde = float("inf")
        min_fde_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort(
                -forecasted_probabilities[k], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t]
                                    for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t]
                               for t in sorted_idx[:max_num_traj]]

        # get min fde, min ade, and index for min fde
        for j in range(len(pruned_trajectories)):
            ade = get_ade(pruned_trajectories[j][:horizon], v[:horizon])
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            mde = get_mde(pruned_trajectories[j][:horizon], v[:horizon])
            if ade < curr_min_ade:
                curr_min_ade = ade
            if fde < curr_min_fde:
                min_fde_idx = j
                curr_min_fde = fde
            if mde < curr_min_mde:
                curr_min_mde = mde

        # get min ade for min fde
        curr_min_ade_for_min_fde = get_ade(
            pruned_trajectories[min_fde_idx][:horizon], v[:horizon])

        # stats
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        min_mde.append(curr_min_mde)
        min_ade_for_min_fde.append(curr_min_ade_for_min_fde)
        n_misses_by_fde.append(curr_min_fde > miss_threshold)
        n_misses_by_mde.append(curr_min_mde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses_by_fde.append(1.0 if curr_min_fde > miss_threshold else (
                1.0 - pruned_probabilities[min_fde_idx]))
            prob_min_ade_for_min_fde.append(
                min(-np.log(pruned_probabilities[min_fde_idx]), -np.log(
                    LOW_PROB_THRESHOLD_FOR_METRICS)) + curr_min_ade_for_min_fde
            )
            prob_min_fde.append(
                min(-np.log(pruned_probabilities[min_fde_idx]), -np.log(
                    LOW_PROB_THRESHOLD_FOR_METRICS)) + curr_min_fde
            )
    metric_results["minADE"+"_"+str(max_guesses)] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"+"_"+str(max_guesses)] = sum(min_fde) / len(min_fde)
    metric_results["minADE(FDE)"+"_"+str(max_guesses)
                   ] = sum(min_ade_for_min_fde) / len(min_ade_for_min_fde)
    metric_results["MR(FDE)"+"_" + str(max_guesses) + "," + str(miss_threshold)+"m"
                   ] = sum(n_misses_by_fde) / len(n_misses_by_fde)
    metric_results["MR(MDE)"+"_" + str(max_guesses) + "," + str(miss_threshold)+"m"
                   ] = sum(n_misses_by_mde) / len(n_misses_by_mde)

    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results["p-MR"] = sum(prob_n_misses_by_fde) / \
            len(prob_n_misses_by_fde)
    return metric_results


################################################################################################
####### The following is adjusted from the latest argoverse API for temp use ###################
################################################################################################

def new_displacement_errors_and_miss_rate(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    max_guesses: int,
    horizon: int,
    miss_threshold: float,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute min fde and ade for each sample.

    Note: Both min_fde and min_ade values correspond to the trajectory which has minimum fde.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        max_guesses: Number of guesses allowed
        horizon: Prediction horizon
        miss_threshold: Distance threshold for the last predicted coordinate
        forecasted_probabilities: Probabilites associated with forecasted trajectories.

    Returns:
        metric_results: Metric values for minADE, minFDE, MR, p-minADE, p-minFDE, p-MR
    """
    metric_results: Dict[str, float] = {}
    min_ade, prob_min_ade = [], []
    min_fde, prob_min_fde = [], []
    n_misses, prob_n_misses = [], []
    for k, v in gt_trajectories.items():
        curr_min_ade = float("inf")
        curr_min_fde = float("inf")
        min_idx = 0
        max_num_traj = min(max_guesses, len(forecasted_trajectories[k]))

        # If probabilities available, use the most likely trajectories, else use the first few
        if forecasted_probabilities is not None:
            sorted_idx = np.argsort([-x for x in forecasted_probabilities[k]], kind="stable")
            # sorted_idx = np.argsort(forecasted_probabilities[k])[::-1]
            pruned_probabilities = [forecasted_probabilities[k][t] for t in sorted_idx[:max_num_traj]]
            # Normalize
            prob_sum = sum(pruned_probabilities)
            pruned_probabilities = [p / prob_sum for p in pruned_probabilities]
        else:
            sorted_idx = np.arange(len(forecasted_trajectories[k]))
        pruned_trajectories = [forecasted_trajectories[k][t] for t in sorted_idx[:max_num_traj]]

        for j in range(len(pruned_trajectories)):
            fde = get_fde(pruned_trajectories[j][:horizon], v[:horizon])
            if fde < curr_min_fde:
                min_idx = j
                curr_min_fde = fde
        curr_min_ade = get_ade(pruned_trajectories[min_idx][:horizon], v[:horizon])
        min_ade.append(curr_min_ade)
        min_fde.append(curr_min_fde)
        n_misses.append(curr_min_fde > miss_threshold)

        if forecasted_probabilities is not None:
            prob_n_misses.append(1.0 if curr_min_fde > miss_threshold else (1.0 - pruned_probabilities[min_idx]))
            prob_min_ade.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_ade
            )
            prob_min_fde.append(
                min(
                    -np.log(pruned_probabilities[min_idx]),
                    -np.log(LOW_PROB_THRESHOLD_FOR_METRICS),
                )
                + curr_min_fde
            )
    metric_results["minADE"] = sum(min_ade) / len(min_ade)
    metric_results["minFDE"] = sum(min_fde) / len(min_fde)
    metric_results["MR"] = sum(n_misses) / len(n_misses)
    if forecasted_probabilities is not None:
        metric_results["p-minADE"] = sum(prob_min_ade) / len(prob_min_ade)
        metric_results["p-minFDE"] = sum(prob_min_fde) / len(prob_min_fde)
        metric_results["p-MR"] = sum(prob_n_misses) / len(prob_n_misses)
    return metric_results

def compute_forecasting_metrics(
    forecasted_trajectories: Dict[int, List[np.ndarray]],
    gt_trajectories: Dict[int, np.ndarray],
    city_names: Dict[int, str],
    max_n_guesses: int,
    horizon: int,
    forecasted_probabilities: Optional[Dict[int, List[float]]] = None,
) -> Dict[str, float]:
    """Compute all the forecasting metrics.

    Args:
        forecasted_trajectories: Predicted top-k trajectory dict with key as seq_id and value as list of trajectories.
                Each element of the list is of shape (pred_len x 2).
        gt_trajectories: Ground Truth Trajectory dict with key as seq_id and values as trajectory of
                shape (pred_len x 2)
        city_names: Dict mapping sequence id to city name.
        max_n_guesses: Number of guesses allowed
        horizon: Prediction horizon
        forecasted_probabilities: Normalized Probabilities associated with each of the forecasted trajectories.

     Returns:
        metric_results: Dictionary containing values for all metrics.
    """
    metric_results = new_displacement_errors_and_miss_rate(
        forecasted_trajectories,
        gt_trajectories,
        max_n_guesses,
        horizon,
        _MISSING_THRESHOLD,
        forecasted_probabilities,
    )
    # metric_results["DAC"] = get_drivable_area_compliance(forecasted_trajectories, city_names, max_n_guesses)

    print("------------------------------------------------")
    print(f"Prediction Horizon : {horizon}, Max #guesses (K): {max_n_guesses}")
    print("------------------------------------------------")
    print(metric_results)
    print("------------------------------------------------")

    return metric_results