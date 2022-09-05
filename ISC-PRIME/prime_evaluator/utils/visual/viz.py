# Tools for Visualization
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Union
from shapely.geometry import LineString


# Visualization parameters
_LINE_WIDTH = {"AGENT": 3.0, "AV": 1.5, "OTHERS": 1.5}
_MARKER_SIZE = {"AGENT": 8, "AV": 6, "OTHERS": 6}
_COLOR = {"AGENT": ["#ECA154", "#d33e4c"],  # Soft Orange, Moderate red
          "AV": "#007672",  # Dark cyan
          "OTHERS": "#d3e8ef",  # Light grayish blue
          }
_ZORDER = {"AGENT": 20, "AV": 15, "OTHERS": 10}

######################################################################################
############################## DEBUG Plot ############################################
######################################################################################

def debug_plot_eles(eles: List,
                    new: bool=False, show: bool = False,
                    pstyle="x", pcolor="red", markersize=7,
                    lstyle="--", lcolor="dimgrey", linewidth=1, lendpoint=True,
                    zorder = 0):
    if new:
        plt.figure(figsize=(8, 7))
        plt.title("DEBUG")
    for id, ele in enumerate(eles):
        # Plot point with its id
        if ele.ndim==1:
            plt.plot(ele[0], ele[1], pstyle, color=pcolor, alpha=1, markersize=markersize, zorder=zorder+1)
            plt.text(ele[0], ele[1], f"{id}")
        # Plot line with its endpoint
        elif ele.ndim==2:
            plt.plot(ele[:, 0], ele[:, 1], lstyle, color=lcolor, alpha=1, linewidth=linewidth, zorder=zorder)
            if lendpoint:
                plt.plot(ele[-1, 0], ele[-1, 1], pstyle, color=pcolor, alpha=1, markersize=markersize, zorder=zorder+1)
                plt.text(ele[-1, 0], ele[-1, 1], f"{id}")
    plt.axis("equal")
    if show:
        plt.show()


def debug_plot_pt(xy: np.ndarray, new: bool=False, show = False,
                  style="X", color="red", alpha=1, markersize=7, zorder=5):
    if new:
        plt.figure(figsize=(8, 7))
        plt.title("DEBUG")
    plt.plot(xy[0], xy[1], style,
             color = color, alpha = alpha, markersize = markersize, zorder = zorder)
    plt.axis("equal")
    if show:
        plt.show()


def debug_plot_line(xy: np.ndarray, new: bool=False, show = False, endpt = False,
                    style="-.", color="dimgrey", alpha=1, linewidth=1, markersize=1, zorder=0):
    if new:
        plt.figure(figsize=(8, 7))
        plt.title("DEBUG")
    plt.plot(xy[:, 0], xy[:, 1], style,
             color=color, alpha=alpha, linewidth=linewidth, markersize=1, zorder=zorder)
    if endpt:
        plt.text(xy[0][0], xy[0][1], "s")
        plt.text(xy[-1][0], xy[-1][1], "end")
    plt.axis("equal")
    if show:
        plt.show()


def debug_plot_centered_circle(loc: Union[np.ndarray, List[float]], radius: float,
                               new: bool=False, show = False,
                               color="red", linewidth=1, linestyle=":", alpha=1, zorder=0):
    if new:
        plt.figure(figsize=(8, 7))
        plt.title("DEBUG")
    circle = plt.Circle((loc[0], loc[1]), radius, fill=False,
                        color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)
    plt.gca().add_artist(circle)
    if show:
        plt.show()

######################################################################################
############################## Plot on axes ##########################################
######################################################################################




def plot_centered_circle(ax: plt.Axes, loc: Union[np.ndarray, List[float]], radius: float,
                         color="red", linewidth=1, linestyle=":", alpha=1, zorder=0):
    circle = plt.Circle((loc[0], loc[1]), radius, fill=False,
                        color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)
    ax.add_artist(circle)


def plot_traj_with_endpoint(ax: plt.Axes, x_cords: np.ndarray, y_cords: np.ndarray, scale: float = 1.0,
                            color="dimgrey",
                            linestyle='-', linewidth=1.0, linealpha=1.0,
                            markstyle='o', markersize=3.0, markeralpha=1.0,
                            zorder=0, label="",
                            ignore_endpoint = False):
    ax.plot(x_cords * scale, y_cords * scale,
            linestyle, alpha=linealpha, label=label,
            color=color, linewidth=linewidth, zorder=zorder)
    if not ignore_endpoint:
        ax.plot(x_cords[-1] * scale, y_cords[-1] * scale,
                markstyle, alpha=markeralpha,
                color=color, markersize=markersize, zorder=zorder)


def plot_lane_centerlines(ax: plt.Axes, centerlines: Union[List[np.ndarray], np.ndarray], scale: float = 1.0,
                          with_rank: bool = False,
                          linestyle="--", color="lightgrey", alpha=1.0, linewidth=1.0, zorder=0):
    """Draw the given 2D centerlin (similar to argoverse-api/argoverse/util/mpl_plotting_utils: visualize_centerline)
    Args:
    """
    rank = 0
    if not isinstance(centerlines, list):
        centerlines = [centerlines]
    for centerline in centerlines:
        ax.plot(centerline[:, 0] * scale, centerline[:, 1] * scale, linestyle, color=color, alpha=alpha, linewidth=linewidth, zorder=zorder)
        if with_rank:
            ax.text(centerline[0][0] * scale, centerline[0][1] * scale, f"${rank}_s$", zorder=zorder+50)
            ax.text(centerline[-1][0] * scale, centerline[-1][1] * scale, f"${rank}_e$", zorder=zorder+50)
            rank += 1


######################################################################################
########################### Direct Visualization #####################################
######################################################################################


def viz_centerline_xy_sample(sample: Dict[str, np.ndarray], scale_revert: bool = False, save_loc:str = None):
    '''
    Visualize single sample
    '''
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    scale_factor = sample['scale'] if scale_revert else 1.0

    ##################### Elements to be visualized #####################
    seq_id = sample['seq_id']
    xy_obs = sample['agent_obs_xy']
    xy_gt = sample['agent_gt_xy']
    nbrs_obs = [sample['nbrs_obs_xy'][nbr_id] for nbr_id in range(sample['nbrs_obs_xy'].shape[0])]
    all_xy_futs = np.vstack([trajs for trajs in sample['agent_futs_xy']])

    ##################### Subfigure 1 #####################
    plot_traj_with_endpoint(ax1, xy_obs[:, 0], xy_obs[:, 1], scale=scale_factor, color="#ECA154", linewidth=3, markersize=9, zorder=15, label="Observed")
    plot_traj_with_endpoint(ax1, xy_gt[:, 0], xy_gt[:, 1], scale=scale_factor, color="#d33e4c", linewidth=3, markersize=9, zorder=20, label="Ground Truth")
    for nbr_xy in nbrs_obs:
        plot_traj_with_endpoint(ax1, nbr_xy[:, 0], nbr_xy[:, 1], scale=scale_factor, color="#d3e8ef", linewidth=3, markersize=9, zorder=5)

    for id, fut in enumerate(all_xy_futs):
        if id==sample['fut_oracle_id']:
            plot_traj_with_endpoint(ax1, fut[:, 0], fut[:, 1], scale=scale_factor, color="#007672", linewidth=3.0, linealpha=1.0, markstyle='X', markersize=6, markeralpha=0.8, zorder=30)
        else:
            plot_traj_with_endpoint(ax1, fut[:, 0], fut[:, 1], scale=scale_factor, color="blue", linewidth=0.2, linealpha=0.5, markstyle='x', markersize=1, markeralpha=0.5, zorder=25)

    plot_lane_centerlines(ax1, [sample['agent_cls'][i] for i in range(sample['agent_cls'].shape[0])], scale=scale_factor, color="grey", linewidth=5, alpha=0.5, zorder=0, with_rank=1)
    trajs_stat = [len(i) for i in sample['agent_futs_xy']]
    if sample['agent_cls_oracle'] is None:
        trajs_stat_str = ''.join(str(num) + '/ ' for i, num in enumerate(trajs_stat))
    else:
        trajs_stat_str = ''.join(str(num) + '*' * (i in sample['agent_cls_oracle']) + '/ ' for i, num in enumerate(trajs_stat))
        plot_lane_centerlines(ax1, [sample['agent_cls'][i] for i in sample['agent_cls_oracle']], scale=scale_factor,
                              color="pink", linewidth=5, alpha=1.0, zorder=1, with_rank=False)

    ##################### Subfigure 2 #####################
    plot_traj_with_endpoint(ax2, xy_obs[:, 0], xy_obs[:, 1], scale=1.0, color="#ECA154", linewidth=2, markersize=9, markstyle='x', zorder=15, label="Observed")
    plot_traj_with_endpoint(ax2, xy_gt[:, 0], xy_gt[:, 1], scale=1.0, color="#d33e4c", linewidth=2, markersize=9, markstyle='x', zorder=20, label="Ground Truth")
    for fut in all_xy_futs:
        plot_traj_with_endpoint(ax2, fut[:, 0], fut[:, 1], scale=1.0, color="blue", linewidth=0.2, linealpha=0.2, markstyle='o', markersize=4, markeralpha=0.2, zorder=10)

    ##################### Other information #####################
    all_xy_futs_diff = (all_xy_futs - xy_gt) * sample['scale']
    all_xy_futs_disp = np.hypot(all_xy_futs_diff[:, :, 0], all_xy_futs_diff[:, :, 1])
    min_fde_id = np.argmin(all_xy_futs_disp[:, -1])
    min_fde = all_xy_futs_disp[min_fde_id, -1]
    argo_ade = all_xy_futs_disp[min_fde_id, :].mean()
    min_ade = np.mean(all_xy_futs_disp, axis=1).min()

    pre_title = "##TEST## " if math.isnan(xy_gt[0][0]) else ""

    ax1.set_title(f"{pre_title}Seq {seq_id}: Trajs={trajs_stat_str}")
    ax2.set_title(f"fdeN={round(min_fde, 2)}   adeN={round(min_ade, 2)}")
    ax1.axis("equal")
    ax2.axis("equal")
    plt.legend()

    if save_loc:
        print(f"Ploting seq-{seq_id}")
        img_file = os.path.join(save_loc, f"{seq_id}.png")
        plt.savefig(img_file)
    else:
        plt.show()