import copy
from dis import dis
import math
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import LineString, Point

from hdmap.hd_map import HDMap
from hdmap.object.lane import Lane
from dataset.pandas_dataset import DATA_DICT

from hdmap.visual.map_vis import draw_lanelet_map
from path_search.visual.path_viz import plot_path, plot_cl, plot_cl_array

from util_dir.geometry import get_angle, normalize_angle


def check_path(path: List[int], hd_map: HDMap) -> bool:
    for idx in range(len(path)-1):
        s_id, e_id = path[idx], path[idx+1]
        if hd_map.road_graph[s_id][e_id]["lane_change"]:
            if nx.has_path(G=hd_map.directed_graph, source=path[0], target=e_id):
                return False

    return True


def find_all_paths(lane_list: List[int], hd_map: HDMap, roundabout: bool = False) -> List[List[List[int]]]:
    # print("Roundabout = {}".format(roundabout))
    path_list = []

    leaves_node = [v for v, d in hd_map.directed_graph.out_degree() if d == 0]
    for lane_id in lane_list:
        if roundabout:
            all_path = nx.all_simple_paths(G=hd_map.road_graph, source=lane_id, target=leaves_node, cutoff=15)
        else:
            all_path = nx.all_simple_paths(G=hd_map.road_graph, source=lane_id, target=leaves_node)
        filter_path = []
        for path in all_path:
            if check_path(path, hd_map):
                filter_path.append(path)
        if len(filter_path) == 0:
            filter_path.append([lane_id])
        path_list.append(filter_path)
    return path_list


def find_directed_paths(lane_list: List[int], hd_map: HDMap) -> List[List[List[int]]]:
    path_list = []

    leaves_node = [v for v, d in hd_map.directed_graph.out_degree() if d == 0]
    for lane_id in lane_list:
        all_path = nx.all_simple_paths(G=hd_map.directed_graph, source=lane_id, target=leaves_node)
        filter_path = []
        for path in all_path:
            filter_path.append(path)

        path_list.append(filter_path)
    return path_list


def get_cl(path: List[int], hd_map: HDMap) -> np.ndarray:
    idx = 0
    cl_list = []
    edge_attribute = nx.get_edge_attributes(hd_map.road_graph, "lane_change")
    while idx < len(path):
        if idx != len(path)-1 and edge_attribute[(path[idx], path[idx+1])]:
            change_list = [idx, idx+1]
            change_idx = idx+1
            while True:
                if change_idx != len(path)-1 and edge_attribute[(path[change_idx], path[change_idx+1])]:
                    change_list.append(change_idx+1)
                    change_idx += 1
                else:
                    break

            start_lane_obj = hd_map.id_lane_dict[path[change_list[0]]]
            end_lane_obj = hd_map.id_lane_dict[path[change_list[-1]]]

            start_ls = LineString(start_lane_obj.centerline_array)
            project_point = Point(end_lane_obj.centerline_array[0])
            start_dist = start_ls.project(project_point)
            start_p = start_ls.interpolate(start_dist+end_lane_obj.lane_length/3)

            start_idx = start_lane_obj.get_idx(dist=start_dist)
            line_list = [(start_lane_obj.centerline_array[i][0], start_lane_obj.centerline_array[i][1]) for i in range(start_idx+1)]
            line_list.append((start_p.x, start_p.y))

            end_ls = LineString(end_lane_obj.centerline_array)
            end_p = end_ls.interpolate(end_ls.length*2/3)
            end_dist = end_lane_obj.get_dist_with_point(target_p=np.array([end_p.x, end_p.y]))
            end_idx = end_lane_obj.get_idx(end_dist)

            line_list.append((end_p.x, end_p.y))
            for i in range(end_idx+1, end_lane_obj.centerline_array.shape[0]):
                line_list.append((end_lane_obj.centerline_array[i][0], end_lane_obj.centerline_array[i][1]))

            cl_list.extend(line_list)
            idx = change_idx + 1
        else:
            lane_obj = hd_map.id_lane_dict[path[idx]]
            line_list = [(lane_obj.centerline_array[i][0], lane_obj.centerline_array[i][1])
                         for i in range(lane_obj.centerline_array.shape[0])]

            cl_list.extend(line_list)
            idx += 1

    return np.asarray(cl_list)


def filter_path_with_speed(path_cl: np.ndarray, v_pos: np.ndarray, speed_limit: float = 80/3.6, time_interval: float = 4):
    cl_ls = LineString(path_cl)
    cur_p = Point(v_pos)
    cur_dist = cl_ls.project(cur_p)
    max_dist = cur_dist + speed_limit * time_interval

    if max_dist > cl_ls.length:
        return path_cl
    else:
        end_p = cl_ls.interpolate(max_dist)

        virtual_lane = Lane(0, 0, 0, "virtual", "virtual", "virtual", False, "virtual")
        virtual_lane.centerline_array = path_cl
        end_idx = virtual_lane.get_idx(max_dist)

        res_array = copy.deepcopy(path_cl[:end_idx+1, :])
        res_array = np.concatenate((res_array, np.array([[end_p.x, end_p.y]])), axis=0)

        return res_array

# def filter_path(all_path_list: List[List[List[int]]], hd_map: HDMap, track_obs_full: np.ndarray, track_id: int,
#                 case_data: pd.DataFrame, max_speed: float = 100/3.6):
#     final_path = []
#     track_obs_xy = track_obs_full[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
#     velocity = track_obs_full[:, [DATA_DICT["vx"], DATA_DICT["vy"]]].astype("float")
#     cur_vel_vec = velocity[-1]
#     cur_pos = track_obs_xy[-1]
#
#     agents = np.unique(case_data["track_id"].values)
#
#     for path_list in all_path_list:
#         for path in path_list:
#             constrained_path = []
#             for lane_id in path:
#                 lane_obj = hd_map.id_lane_dict[lane_id]


def filter_lane(lane_list: List[int], v_heading: float, v_pos: np.ndarray, id_lane_dict: Dict[int, Lane], hd_map: HDMap):
    filter_lane_list = []

    v_point = Point(v_pos)
    for lane_id in lane_list:
        lane_obj = id_lane_dict[lane_id]
        cl_ls = LineString(lane_obj.centerline_array)
        dist = cl_ls.project(v_point)

        start_idx = lane_obj.get_idx(dist)

        direction_vec = lane_obj.centerline_array[start_idx+1]-lane_obj.centerline_array[start_idx]
        lane_angle = get_angle(vec_a=np.array([1, 0]), vec_b=direction_vec)
        angle_diff = normalize_angle(v_heading - lane_angle)

        if abs(angle_diff) < math.pi*2/3:
            filter_lane_list.append(lane_id)

    return filter_lane_list


def filter_cl_with_distance(cl_list: List[np.ndarray], track_obs_array: np.ndarray):
    min_dist = 10e9
    min_cl = None

    for cl in cl_list:
        cl_ls = LineString(coordinates=cl)

        project_p = []
        for i in range(track_obs_array.shape[0]):
            point = Point(track_obs_array[i])
            dist = cl_ls.project(point)
            nearest_p = cl_ls.interpolate(dist)
            project_p.append((nearest_p.x, nearest_p.y))
        project_p_array = np.asarray(project_p)

        diff = track_obs_array-project_p_array

        # test = np.hypot(diff[:, 0], diff[:, 1])
        # print(diff[:, 0])
        # print(diff[:, 1])
        nl_dist = np.sum(np.hypot(diff[:, 0], diff[:, 1]))

        if nl_dist < min_dist:
            min_dist = nl_dist
            min_cl = copy.deepcopy(cl)

    return min_cl


def filter_cl_with_end_point(cl_list: List[np.ndarray]):
    cl_dict = {}
    for cl in cl_list:
        if len(cl_dict) == 0:
            cl_dict[(cl[-1][0], cl[-1][1])] = [cl]
        else:
            cl_end_p = copy.deepcopy(cl[-1])

            is_add = False
            for end_p, cl_set in cl_dict.items():
                diff_array = np.asarray(end_p)-cl_end_p
                dist = np.hypot(diff_array[0], diff_array[1])

                if dist <= 3:
                    cl_set.append(cl)
                    is_add = True
                    break
            
            if not is_add:
                cl_dict[(cl_end_p[0], cl_end_p[1])] = [cl]
    
    return cl_dict


def filter_cl_with_angle(cl_list: List[np.ndarray], track_obs_xy: np.ndarray):
    filter_cl_list = []

    for cl in cl_list:
        cl_ls = LineString(coordinates=cl)

        is_add = True
        for idx in range(track_obs_xy.shape[0]):
            point = Point(track_obs_xy[idx])
            dist = cl_ls.project(point)
            project_point = cl_ls.interpolate(dist)

            if dist+2 <= cl_ls.length:
                far_point = cl_ls.interpolate(dist+2)
                direct_array = np.array([far_point.x, far_point.y]) - np.array([project_point.x, project_point.y])
            else:
                close_dist = dist-2 if dist-2 >= 0 else 0
                close_point = cl_ls.interpolate(close_dist)
                direct_array = np.array([project_point.x, project_point.y]) - np.array([close_point.x, close_point.y])
            if idx != track_obs_xy.shape[0]-1:
                traj_array = track_obs_xy[idx+1]-track_obs_xy[idx]
            else:
                traj_array = track_obs_xy[idx] - track_obs_xy[idx-1]
            
            if abs(traj_array[0]) < 1e-6 and abs(traj_array[1]) < 1e-6:
                continue
            lane_angle = get_angle(direct_array, traj_array)

            if abs(normalize_angle(lane_angle)) > math.pi/2:
                is_add = False
                break
        
        if is_add:
            filter_cl_list.append(cl)

    return filter_cl_list


def filter_cl(cl_list: List[np.ndarray], track_obs_array: np.ndarray):
    cl_dict = filter_cl_with_end_point(cl_list)

    filter_cl = []

    for _, cl_set in cl_dict.items():
        min_cl = filter_cl_with_distance(cl_set, track_obs_array)

        filter_cl.append(min_cl)

    filter_after_angle = filter_cl_with_angle(filter_cl, track_obs_array)

    if len(filter_after_angle) == 0:
        return filter_cl
    else:
        return filter_after_angle
    # return filter_cl


def delete_repeat_point(cl: np.ndarray):
    resonable_cl = []

    for i in range(cl.shape[0]):
        if i == 0:
            resonable_cl.append((cl[i][0], cl[i][1]))
        else:
            if abs(cl[i][0]-resonable_cl[-1][0]) < 1e-6 and abs(cl[i][1]-resonable_cl[-1][1]) < 1e-6:
                continue
            else:
                resonable_cl.append((cl[i][0], cl[i][1]))
    return np.array(resonable_cl)


def path_search_rule(track_obs_xy: np.ndarray, track_obs_heading: np.ndarray,
                     case_data: pd.DataFrame, track_id: int, hd_map: HDMap, roundabout: bool = False):
    # track_obs_xy = track_obs_full[:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
    # track_obs_heading = track_obs_full[:, [DATA_DICT["psi_rad"]]].astype("float")

    # print(np.unique(case_data["track_id"].values))
    # print(track_obs_xy)
    # print(track_obs_heading)

    # axes = plt.subplot(111)
    # axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)
    # axes.plot(track_obs_xy[:, 0], track_obs_xy[:, 1], color="purple")
    # axes.scatter(track_obs_xy[-1, 0], track_obs_xy[-1, 1], color="purple", marker="o", s=25)

    lane_list = hd_map.find_lanelet(pos=track_obs_xy[0])
    # filter_lane_list = filter_lane(lane_list=lane_list,
    #                                v_heading=track_obs_heading[-1],
    #                                v_pos=track_obs_xy[-1],
    #                                id_lane_dict=hd_map.id_lane_dict,
    #                                hd_map=hd_map)

    # print("{}/{}".format(len(lane_list), len(filter_lane_list)))
    all_path_list = find_all_paths(lane_list=lane_list, hd_map=hd_map, roundabout=roundabout)

    cl_list = []
    for path_list in all_path_list:
        for path in path_list:
            # print(path)
            # axes = plot_path(path, hd_map.id_lane_dict, axes=axes, color="green")
            # axes = plot_cl(path, hd_map.id_lane_dict, axes=axes, color="gray")
            cl = get_cl(path, hd_map)
            speed_limit = hd_map.id_lane_dict[path[0]].get_speed_limit()
            cl = filter_path_with_speed(path_cl=cl,
                                        v_pos=track_obs_xy[0],
                                        speed_limit=speed_limit if speed_limit is not None else 80/3.6,
                                        time_interval=4)
            cl_list.append(delete_repeat_point(cl))
    #         axes = plot_cl_array(cl, axes)
    #     print("-----------------------------------------------")
    # print(lane_list)
    #
    # plt.show()

    min_cl = filter_cl(cl_list, track_obs_xy)

    return min_cl