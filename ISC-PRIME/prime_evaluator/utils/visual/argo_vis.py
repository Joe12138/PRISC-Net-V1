import os
import math
import bisect
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, List, Tuple
from scipy.special import softmax
from prime_evaluator.utils.config import (
    _MAX_VIS_GUESSES_NUM
)
from prime_evaluator.utils.visual.viz import (
    plot_traj_with_endpoint,
    plot_lane_centerlines
)
from prime_evaluator.utils.calcu_utils import calcu_min_ade_fde
from prime_evaluator.utils.util import init_csv_stat, save_csv_stat
from prime_evaluator.data_type.centerline_xy_dataset import trans_back_sample_batched

from prime_evaluator.models.traj_classifier import get_probability
from prime_evaluator.models.lane_classifier import get_cond_probability

from hdmap.hd_map import HDMap
from hdmap.visual.map_vis import draw_lanelet_map


PLOT_FDE_NUMS = [5.0, 10.0, 15.0, 20.0]    # if plot option is set, cases with fde under these interval would be draw

class ArgoVis:
    def __init__(self, args):
        self.args = args
        self.save_loc = os.path.abspath(os.path.join(self.args.ckptdir, self.args.subdir or self.args.mode, "normal"))
        self.save_fde_locs = [os.path.abspath(os.path.join(self.args.ckptdir, self.args.subdir or self.args.mode, f"fde_{i}")) for i in PLOT_FDE_NUMS]
        self.csv_file = os.path.abspath(os.path.join(self.args.ckptdir, self.args.subdir or self.args.mode, "visual_stat.csv"))
        if self.args.plot:
            for loc in self.save_fde_locs:
                os.makedirs(loc, exist_ok=True)
            os.makedirs(self.save_loc, exist_ok=True)  # Save images out of the PLOT_FDE_NUMS
            if not os.path.exists(self.csv_file):
                init_csv_stat(self.csv_file, ["sequence", "ade_1", "fde_1", "ade_6", "fde_6"])  # Save statistic

    def render_and_save(self, output, sample_batched, processed_output=None, max_n=16):
        probs = self.output_to_probability(output, sample_batched)
        self.render_batch_preds_scene(sample_batched, processed_output, max_n=max_n, probability=probs, save_img=True)

    def render(self, output, sample_batched, processed_output=None, max_n=16):
        """
        Render the output and sample batched to image
        :param output: Original output produced by the network
        """
        img_list = []
        probs = self.output_to_probability(output, sample_batched)
        figs = self.render_batch_preds(sample_batched, processed_output, max_n=max_n, probability=probs, save_img=False)

        for fig in figs:
            fig.canvas.draw()
            data = np.fromstring(fig.canvas.tostring_rgb(),
                                 dtype=np.uint8, sep='')
            data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img_list.append(data)
        # close all the existing figures
        for fig in figs:
            plt.close(fig)
        img = np.array(img_list)
        img = img.transpose((0, 3, 1, 2))
        return img

    def output_to_probability(self, output, sample_batched):
        if self.args.net_name == 'traj_classifier' or self.args.net_name == "vector_net_classifier":
            probability = get_probability(sample_batched['cls_start_end_idx'],
                                          sample_batched['trajs_start_end_idx'],
                                          traj_scores = output.cpu().detach().numpy())

        elif self.args.net_name == 'lane_classifier':
            lane_scores, traj_scores = output
            probability = get_cond_probability(sample_batched['cls_start_end_idx'],
                                               sample_batched['trajs_start_end_idx'],
                                               lane_scores = lane_scores.cpu().detach().numpy(),
                                               traj_scores = traj_scores.cpu().detach().numpy())
        else:
            assert False, f"no viz function for {self.args.net_name}"
        return probability

    def render_batch_preds_scene(self,
                                 sample_batched: Dict[str, Any],
                                 processed_output,
                                 max_n,
                                 probability: np.ndarray = None,
                                 save_img=False):
        figs = []
        num_samples = min(len(sample_batched["seq_ids"]), max_n)
        sample_batched = trans_back_sample_batched(sample_batched)

        map_path_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps"

        for idx in range(num_samples):
            if save_img:
                if idx == 0:
                    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))
            else:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 8))

            seq_id = sample_batched["seq_ids"][idx]

            seq_list = seq_id.split("_")
            scene_name = f"{seq_list[1]}_{seq_list[2]}_{seq_list[3]}_{seq_list[4]}"
            hd_map = HDMap(osm_file_path=f"{map_path_prefix}/{scene_name}.osm")
            ax = draw_lanelet_map(laneletmap=hd_map.lanelet_map)

            cls_start_end = sample_batched['cls_start_end_idx'][idx]
            nbrs_start_end = sample_batched['nbrs_start_end_idx'][idx]
            futs_start_end = [sample_batched['trajs_start_end_idx'][cls_start_end[0]][0],
                              sample_batched['trajs_start_end_idx'][cls_start_end[1] - 1][1]]
            ##################### Elements to be visualized #####################
            agent_cls_list = [sample_batched['agent_cls'][i] for i in range(cls_start_end[0], cls_start_end[1])]
            nbr_obs_xy_list = [sample_batched['nbrs_obs_xy'][i] for i in range(nbrs_start_end[0], nbrs_start_end[1])]
            agent_obs_xy = sample_batched['agent_obs_xy'][idx]
            agent_gt_xy = sample_batched['agent_gt_xy'][idx]
            agent_fut_xy_list = [sample_batched['agent_futs_xy'][i] for i in
                                 range(futs_start_end[0], futs_start_end[1])]

            pred_trajectories = processed_output[seq_id]

            minFDE_1, minADE_1, _ = calcu_min_ade_fde(pred_trajectories[[0]],
                                                      agent_gt_xy)  # The best trajectory is in the first location
            minFDE_6, minADE_6, _ = calcu_min_ade_fde(pred_trajectories, agent_gt_xy)

            if save_img and (self.args.plot == 'fde' and minFDE_6 < PLOT_FDE_NUMS[0]):
                continue  # Ignore the case if fde < PLOT_FDE_NUMS[0] under 'fde' plot mode

            for nbr_obs in nbr_obs_xy_list:
                plot_traj_with_endpoint(ax, nbr_obs[:, 0], nbr_obs[:, 1], scale=1.0, color="#d3e8ef", linewidth=2, markersize=6, zorder=5)
            plot_traj_with_endpoint(ax, agent_obs_xy[:, 0], agent_obs_xy[:, 1], scale=1.0, color="#ECA154", linewidth=3, markersize=9, zorder=15, label="Observed")
            plot_traj_with_endpoint(ax, agent_gt_xy[:, 0], agent_gt_xy[:, 1], scale=1.0, color="#d33e4c", linewidth=3, markersize=9, zorder=25, label="Ground Truth")
            for agent_fut in agent_fut_xy_list:
                plot_traj_with_endpoint(ax, agent_fut[:, 0], agent_fut[:, 1], scale=1.0, color="blue", linewidth=0.2, linealpha=0.2, markstyle='x', markersize=1.5, markeralpha=0.2, zorder=20)

            ax.set_title(f"Seq {str(seq_id)} || (K=1) ade={minADE_1:.2f} fde={minFDE_1:.2f} (K=6) ade={minADE_6:.2f} fde={minFDE_6:.2f}")
            for pred_id, pred in enumerate(pred_trajectories):
                plot_traj_with_endpoint(ax, pred[:, 0], pred[:, 1], scale=1.0, color="#007672", linewidth=2.0, linealpha=0.5, markstyle='o', markersize=6, markeralpha=0.7, zorder=30)
                # ax.text(pred[-1, 0], pred[-1, 1], f"p{pred_id}", zorder=31)

            if save_img:
                save_csv_stat(self.csv_file, [seq_id, minADE_1, minFDE_1, minADE_6, minFDE_6])
                if math.isnan(minFDE_6):
                    # For test set
                    plt.savefig(os.path.join(self.save_loc, f"{seq_id}.png"))
                else:
                    # Save by their fde value, the others to the normal location.
                    loc_id = bisect.bisect_right(PLOT_FDE_NUMS, minFDE_6)
                    if loc_id > 0:
                        plt.savefig(os.path.join(self.save_fde_locs[loc_id - 1], f"{seq_id}.png"))
                    elif self.args.plot == 'all':
                        plt.savefig(os.path.join(self.save_loc, f"{seq_id}.png"))
                ax.cla()
                # ax2.cla()
                # cb.remove()  # Clear to reuse the figure
            else:
                figs.append(fig)  # To be fed to tensorboard

        if save_img:
            plt.close(fig)

        return figs

    def render_batch_preds(self,
                           sample_batched: Dict[str, Any],
                           processed_output, max_n,
                           probability: np.ndarray = None,
                           save_img = False):
        """
        Shared function for rendering a batch of sample to a list of matplotlib figures
        here sample_batched is already in CPU, and will be transformed back to its original coordinate
        """
        figs = []
        num_samples = min(len(sample_batched['seq_ids']), max_n)
        sample_batched = trans_back_sample_batched(sample_batched)

        for idx in range(num_samples):
            if save_img:
                # use only 1 figure when all figures are saved
                if idx==0:
                    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
            else:
                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

            seq_id = sample_batched['seq_ids'][idx]
            cls_start_end = sample_batched['cls_start_end_idx'][idx]
            nbrs_start_end = sample_batched['nbrs_start_end_idx'][idx]
            futs_start_end = [sample_batched['trajs_start_end_idx'][cls_start_end[0]][0], sample_batched['trajs_start_end_idx'][cls_start_end[1] - 1][1]]
            ##################### Elements to be visualized #####################
            agent_cls_list = [sample_batched['agent_cls'][i] for i in range(cls_start_end[0], cls_start_end[1])]
            nbr_obs_xy_list = [sample_batched['nbrs_obs_xy'][i] for i in range(nbrs_start_end[0], nbrs_start_end[1])]
            agent_obs_xy = sample_batched['agent_obs_xy'][idx]
            agent_gt_xy = sample_batched['agent_gt_xy'][idx]
            agent_fut_xy_list = [sample_batched['agent_futs_xy'][i] for i in range(futs_start_end[0], futs_start_end[1])]

            pred_trajectories = processed_output[seq_id]
            vis_guesses_num = min(_MAX_VIS_GUESSES_NUM, pred_trajectories.shape[0])
            # # Here the probs belong to the top ranked trajectoried (rather than those are filtered)
            # topN_p = sorted(probability[futs_start_end[0]:futs_start_end[1]].squeeze(1), reverse=True)[:vis_guesses_num]
            # topN_probs = topN_p / np.sum(topN_p)
            # prob_text = " / ".join([f"{p*100:.1f}%" for p in topN_probs])
            minFDE_1, minADE_1, _ = calcu_min_ade_fde(pred_trajectories[[0]], agent_gt_xy)  # The best trajectory is in the first location
            minFDE_6, minADE_6, _ = calcu_min_ade_fde(pred_trajectories, agent_gt_xy)

            if save_img and (self.args.plot=='fde' and minFDE_6<PLOT_FDE_NUMS[0]):
                continue    # Ignore the case if fde < PLOT_FDE_NUMS[0] under 'fde' plot mode

            ##################### Subfigure 1 #####################
            plot_lane_centerlines(ax1, centerlines=agent_cls_list, scale=1.0, color="grey", linewidth=5, alpha=0.5, zorder=0, with_rank=True)
            for nbr_obs in nbr_obs_xy_list:
                plot_traj_with_endpoint(ax1, nbr_obs[:, 0], nbr_obs[:, 1], scale=1.0, color="#d3e8ef", linewidth=2, markersize=6, zorder=5)
            plot_traj_with_endpoint(ax1, agent_obs_xy[:, 0], agent_obs_xy[:, 1], scale=1.0, color="#ECA154", linewidth=3, markersize=9, zorder=15, label="Observed")
            plot_traj_with_endpoint(ax1, agent_gt_xy[:, 0], agent_gt_xy[:, 1], scale=1.0, color="#d33e4c", linewidth=3, markersize=9, zorder=25, label="Ground Truth")
            for agent_fut in agent_fut_xy_list:
                plot_traj_with_endpoint(ax1, agent_fut[:, 0], agent_fut[:, 1], scale=1.0, color="blue", linewidth=0.2, linealpha=0.2, markstyle='x', markersize=1.5, markeralpha=0.2, zorder=20)

            ax1.set_title(f"Seq {str(seq_id)} || (K=1) ade={minADE_1:.2f} fde={minFDE_1:.2f} (K=6) ade={minADE_6:.2f} fde={minFDE_6:.2f}")
            for pred_id, pred in enumerate(pred_trajectories):
                plot_traj_with_endpoint(ax1, pred[:, 0], pred[:, 1], scale=1.0, color="#007672", linewidth=2.0, linealpha=0.5, markstyle='o', markersize=6, markeralpha=0.7, zorder=30)
                # ax.text(pred[-1, 0], pred[-1, 1], f"p{pred_id}", zorder=31)

            ##################### Subfigure 2 #####################
            plot_traj_with_endpoint(ax2, agent_obs_xy[:, 0], agent_obs_xy[:, 1], scale=1.0, color="#ECA154", linewidth=1, markersize=9, markstyle='x', zorder=5, label="Observed")
            plot_traj_with_endpoint(ax2, agent_gt_xy[:, 0], agent_gt_xy[:, 1], scale=1.0, color="#d33e4c", linewidth=1, markersize=9, markstyle='x', zorder=15, label="Ground Truth")
            for agent_fut in agent_fut_xy_list:
                plot_traj_with_endpoint(ax2, agent_fut[:, 0], agent_fut[:, 1], scale=1.0, color="blue", linewidth=0.1, linealpha=0.1, markstyle='o', markersize=1.5, markeralpha=0.2, zorder=20, ignore_endpoint=True)
            prob_c = probability[futs_start_end[0]:futs_start_end[1]]
            prob_s = (prob_c / max(prob_c)) * 36 + 4
            probability_obj = ax2.scatter(x=[fut[-1, 0] for fut in agent_fut_xy_list], y=[fut[-1, 1] for fut in agent_fut_xy_list],
                                          c=prob_c, cmap='coolwarm', s=prob_s, alpha=0.6, zorder=30)    # winter / viridis
            cb = plt.colorbar(mappable=probability_obj, ax=ax2)
            ax2.set_title(f"Seq {str(seq_id)} || Distribution")

            ##################### Options #####################
            ax1.axis("equal")
            ax2.axis("equal")
            fig.tight_layout()

            ##################### Others #####################
            if save_img:
                save_csv_stat(self.csv_file, [seq_id, minADE_1, minFDE_1, minADE_6, minFDE_6])
                if math.isnan(minFDE_6):
                    # For test set
                    plt.savefig(os.path.join(self.save_loc, f"{seq_id}.png"))
                else:
                    # Save by their fde value, the others to the normal location.
                    loc_id = bisect.bisect_right(PLOT_FDE_NUMS, minFDE_6)
                    if loc_id > 0:
                        plt.savefig(os.path.join(self.save_fde_locs[loc_id-1], f"{seq_id}.png"))
                    elif self.args.plot=='all':
                        plt.savefig(os.path.join(self.save_loc, f"{seq_id}.png"))
                ax1.cla()
                ax2.cla()
                cb.remove()     # Clear to reuse the figure
            else:
                figs.append(fig)    # To be fed to tensorboard

        if save_img:
            plt.close(fig)

        return figs