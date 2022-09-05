import copy
import math
import numpy as np
import pandas as pd
import networkx as nx

from typing import List
from shapely.geometry import LineString, Point

from hdmap.hd_map import HDMap
from util_dir.geometry import normalize_angle, get_angle
from hdmap.object.lane import Lane

import matplotlib.pyplot as plt


def check_path(path: List[int], hd_map: HDMap) -> bool:
    for idx in range(len(path)-1):
        s_id, e_id = path[idx], path[idx+1]
        if hd_map.road_graph[s_id][e_id]["lane_change"]:
            if nx.has_path(G=hd_map.directed_graph, source=path[0], target=e_id):
                return False
    return True


def find_all_paths(lane_list: List[int], hd_map: HDMap, roundabout: bool = False) -> List[List[List[int]]]:
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


def get_cl(path: List[int], hd_map: HDMap) -> np.ndarray:
    idx = 0
    cl_list = []

    edge_attribute = nx.get_edge_attributes(hd_map.road_graph, "lane_change")
    while idx < len(path):
        if idx != len(path) - 1 and edge_attribute[(path[idx], path[idx + 1])]:
            change_list = [idx, idx + 1]
            change_idx = idx + 1
            while True:
                if change_idx != len(path) - 1 and edge_attribute[(path[change_idx], path[change_idx + 1])]:
                    change_list.append(change_idx + 1)
                    change_idx += 1
                else:
                    break

            start_lane_obj = hd_map.id_lane_dict[path[change_list[0]]]
            end_lane_obj = hd_map.id_lane_dict[path[change_list[-1]]]

            start_ls = LineString(start_lane_obj.centerline_array)
            project_point = Point(end_lane_obj.centerline_array[0])
            start_dist = start_ls.project(project_point)
            start_p = start_ls.interpolate(start_dist + end_lane_obj.lane_length / 3)

            start_idx = start_lane_obj.get_idx(dist=start_dist)
            line_list = [(start_lane_obj.centerline_array[i][0], start_lane_obj.centerline_array[i][1]) for i in
                         range(start_idx + 1)]
            line_list.append((start_p.x, start_p.y))

            end_ls = LineString(end_lane_obj.centerline_array)
            end_p = end_ls.interpolate(end_ls.length * 2 / 3)
            end_dist = end_lane_obj.get_dist_with_point(target_p=np.array([end_p.x, end_p.y]))
            end_idx = end_lane_obj.get_idx(end_dist)

            line_list.append((end_p.x, end_p.y))
            for i in range(end_idx + 1, end_lane_obj.centerline_array.shape[0]):
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


def filter_cl_with_end_point(cl_list: List[np.ndarray]):
    endp_idx_dict = {}
    for idx, cl in enumerate(cl_list):
        if len(endp_idx_dict) == 0:
            endp_idx_dict[(cl[-1][0], cl[-1][1])] = [idx]
        else:
            cl_end_p = copy.deepcopy(cl[-1])

            is_add = False
            for end_p, idx_list in endp_idx_dict.items():
                diff_array = np.asarray(end_p) - cl_end_p
                dist = np.hypot(diff_array[0], diff_array[1])

                if dist <= 3:
                    idx_list.append(idx)
                    is_add = True
                    break

            if not is_add:
                endp_idx_dict[(cl_end_p[0], cl_end_p[1])] = [idx]

    return endp_idx_dict


def filter_cl_with_distance(cl_idx_list: List[int], track_obs_array: np.ndarray, cl_list: List[np.ndarray]):
    min_dist = 10e9
    min_cl_idx = None

    for cl_idx in cl_idx_list:
        cl_ls = LineString(coordinates=cl_list[cl_idx])

        project_p = []
        for i in range(track_obs_array.shape[0]):
            point = Point(track_obs_array[i])
            dist = cl_ls.project(point)
            nearest_p = cl_ls.interpolate(dist)
            project_p.append((nearest_p.x, nearest_p.y))
        project_p_array = np.asarray(project_p)

        diff = track_obs_array - project_p_array

        nl_dist = 0
        for i in range(diff.shape[0]):
            nl_dist += math.sqrt(diff[i][0]**2+diff[i][1]**2)

        # nl_dist = np.sum(np.hypot(diff[:, 0], diff[:, 1]))

        if nl_dist < min_dist:
            min_dist = nl_dist
            min_cl_idx = copy.deepcopy(cl_idx)

    return min_cl_idx


def filter_cl_with_angle(cl_list: List[np.ndarray], path_list: List[List[int]], track_obs_xy: np.ndarray):
    filter_cl_list = []
    filter_path_list = []

    for cl_idx, cl in enumerate(cl_list):
        cl_ls = LineString(coordinates=cl)

        is_add = True
        for index in range(track_obs_xy.shape[0]):
            point = Point(track_obs_xy[index])
            dist = cl_ls.project(point)
            project_point = cl_ls.interpolate(dist)

            if dist+2 <= cl_ls.length:
                far_point = cl_ls.interpolate(dist+2)
                direct_array = np.array([far_point.x, far_point.y])-np.array([project_point.x, project_point.y])
            else:
                close_dist = dist-2 if dist-2 >= 0 else 0
                close_point = cl_ls.interpolate(close_dist)
                direct_array = np.array([project_point.x, project_point.y])-np.array([close_point.x, close_point.y])

            if index != track_obs_xy.shape[0]-1:
                traj_array = track_obs_xy[index+1]-track_obs_xy[index]
            else:
                traj_array = track_obs_xy[index]-track_obs_xy[index-1]

            if abs(traj_array[0]) < 1e-6 and abs(traj_array[1]) < 1e-6:
                continue
            lane_angle = get_angle(direct_array, traj_array)

            if abs(normalize_angle(lane_angle)) > math.pi / 2:
                is_add = False
                break

        if is_add:
            filter_cl_list.append(cl)
            filter_path_list.append(path_list[cl_idx])
    return filter_cl_list, filter_path_list


def filter_cl(cl_list: List[np.ndarray], path_list: List[List[int]], track_obs_array: np.ndarray):
    # axes = plt.subplot(111)
    # for cl in cl_list:
    #     axes.plot(cl[:, 0], cl[:, 1], color="k")
    
    # plt.show()
    # plt.cla()
    # plt.clf()
    endp_idx_dict = filter_cl_with_end_point(cl_list)

    filter_cl = []
    filter_path = []

    for _, cl_idx_list in endp_idx_dict.items():
        min_cl_idx = filter_cl_with_distance(cl_idx_list=cl_idx_list,
                                             track_obs_array=track_obs_array,
                                             cl_list=cl_list)

        filter_cl.append(copy.deepcopy(cl_list[min_cl_idx]))
        filter_path.append(copy.deepcopy(path_list[min_cl_idx]))

    # axes = plt.subplot(111)
    # for cl in filter_cl:
    #     axes.plot(cl[:, 0], cl[:, 1], color="k")
    
    # plt.show()
    # plt.cla()
    # plt.clf()

    # filter_cl_by_angle, filter_path_by_angle = filter_cl_with_angle(cl_list=filter_cl,
    #                                                                 path_list=filter_path,
    #                                                                 track_obs_xy=track_obs_array)

    # axes = plt.subplot(111)
    # for cl in filter_cl_by_angle:
    #     axes.plot(cl[:, 0], cl[:, 1], color="k")
    
    # plt.show()
    # plt.cla()
    # plt.clf()

    # if len(filter_cl_by_angle) == 0:
    #     return filter_cl, filter_path
    # else:
    #     return filter_cl_by_angle, filter_path_by_angle
    return filter_cl, filter_path


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


def binary_search(target_ls: LineString, start: int, end: int, hd_map: HDMap, path: List[int]):
    if start > end:
        return None

    mid = math.ceil(start+(end-start)/2)
    mid_cl = get_cl(path=path[:mid], hd_map=hd_map)
    mid_cl_ls = LineString(coordinates=mid_cl)

    if mid_cl_ls.length > target_ls.length:
        if mid == 1:
            return mid
        else:
            last_mid_cl = get_cl(path=path[:mid-1], hd_map=hd_map)
            last_mid_cl_ls = LineString(coordinates=last_mid_cl)

            if last_mid_cl_ls.length < target_ls.length:
                return mid
        return binary_search(target_ls=target_ls, start=start, end=mid-1, path=path, hd_map=hd_map)
    else:
        if mid == len(path):
            return mid
        else:
            next_mid_cl = get_cl(path=path[:mid+1], hd_map=hd_map)
            next_mid_cl_ls = LineString(coordinates=next_mid_cl)
            if next_mid_cl_ls.length > target_ls.length:
                return mid+1

        return binary_search(target_ls=target_ls, start=mid+1, end=end, path=path, hd_map=hd_map)


def get_corresponding_path(cl: np.ndarray, path: List[int], hd_map: HDMap):
    cl_ls = LineString(coordinates=cl)

    if len(path) == 1:
        return path

    idx_res = binary_search(target_ls=cl_ls, start=0, end=len(path), hd_map=hd_map, path=path)

    return path[:idx_res]


def path_search_rule(track_obs_xy: np.ndarray, track_obs_yaw: np.ndarray, case_data: pd.DataFrame, track_id: int,
                     hd_map: HDMap, roundabout: bool = False):
    lane_list = hd_map.find_lanelet(pos=track_obs_xy[0])

    all_path_list = find_all_paths(lane_list=lane_list, hd_map=hd_map, roundabout=roundabout)

    cl_list = []
    cor_path_list = []
    for path_list in all_path_list:
        for path in path_list:
            cl = get_cl(path, hd_map)
            speed_limit = hd_map.id_lane_dict[path[0]].get_speed_limit()
            cl = filter_path_with_speed(path_cl=cl,
                                        v_pos=track_obs_xy[0],
                                        speed_limit=speed_limit if speed_limit is not None else 80 / 3.6,
                                        time_interval=4)

            corresponding_path = get_corresponding_path(cl=cl, path=path, hd_map=hd_map)
            cl_list.append(delete_repeat_point(cl))
            cor_path_list.append(corresponding_path)
    
    # DEBUG
    # return cl_list, cor_path_list

    filter_cl_list, filter_path_list = filter_cl(cl_list=cl_list,
                                                 path_list=cor_path_list,
                                                 track_obs_array=track_obs_xy)
    return filter_cl_list, filter_path_list


def get_path_set(track_obs_xy: np.ndarray, hd_map: HDMap, roundabout: bool = False):
    lane_list = hd_map.find_lanelet(pos=track_obs_xy[0])

    all_path_list = find_all_paths(lane_list=lane_list, hd_map=hd_map, roundabout=roundabout)
    
    path_set = set()
    for path_list in all_path_list:
        for path in path_list:
            for lane_id in path:
                path_set.add(lane_id)
    
    return path_set
