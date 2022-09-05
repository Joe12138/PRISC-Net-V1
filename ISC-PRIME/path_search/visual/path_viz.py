import copy
from typing import List, Dict
from cv2 import flip

import numpy as np
from shapely.geometry import LineString, Point
import matplotlib.axes
from hdmap.object.lane import Lane
from hdmap.hd_map import HDMap


def plot_path(path: List[int], id_lane_dict: Dict[int, Lane], axes: matplotlib.axes, color: str = "orange"):
    for i in range(len(path)-1):
        s_id, e_id = path[i], path[i+1]

        s_lane_obj, e_lane_obj = id_lane_dict[s_id], id_lane_dict[e_id]
        s_cl_ls, e_cl_ls = LineString(s_lane_obj.centerline_array), LineString(e_lane_obj.centerline_array)

        s_p, e_p = s_cl_ls.interpolate(s_cl_ls.length/2), e_cl_ls.interpolate(e_cl_ls.length/2)

        axes.annotate("", xy=(e_p.x, e_p.y), xytext=(s_p.x, s_p.y),
                      arrowprops=dict(arrowstyle="->", color=color, linestyle="-", linewidth=2, zorder=100), alpha=0.7)

    return axes


def plot_cl(path: List[int], id_lane_dict: Dict[int, Lane], axes:matplotlib.axes, color: str = "gray"):
    cl_array = None
    for i in range(len(path)):
        s_id = path[i]
        s_lane_obj = id_lane_dict[s_id]
        if i == 0:
            cl_array = copy.deepcopy(s_lane_obj.centerline_array)
        else:
            cl_array = np.concatenate((cl_array, copy.deepcopy(s_lane_obj.centerline_array)), axis=0)
    axes.plot(cl_array[:, 0], cl_array[:, 1], color=color, linestyle="--")
    return axes


def plot_cl_array(cl: np.ndarray, axes: matplotlib.axes, color: str= "gray"):
    axes.plot(cl[:, 0], cl[:, 1], color=color, linestyle="--")
    return axes

def plot_path_area(cl: np.ndarray, start_point: np.ndarray, path: List[int], id_lane_dict: Dict[int, Lane], hd_map: HDMap, axes: matplotlib.axes):
    # start_point = cl[9]
    start_lane_list = hd_map.find_lanelet(pos=start_point)
    start_symbol = True
    for idx, lane_id in enumerate(path):
        if lane_id in start_lane_list:
            start_symbol = False
            start_lane_obj = id_lane_dict[lane_id]
            start_point_obj = Point(start_point)

            left_cl = LineString(start_lane_obj.left_way_array)
            right_cl = LineString(start_lane_obj.right_way_array)
            left_dist = left_cl.project(start_point_obj)
            right_dist = right_cl.project(start_point_obj)

            x_array = []
            y_array = []

            dist = left_dist
            while dist < left_cl.length:
                left_p = left_cl.interpolate(dist)
                x_array.append(left_p.x)
                y_array.append(left_p.y)
                dist += 1
            left_p = left_cl.interpolate(left_cl.length)
            x_array.append(left_p.x)
            y_array.append(left_p.y)

            dist = right_cl.length
            while dist > right_dist:
                right_p = right_cl.interpolate(dist)
                x_array.append(right_p.x)
                y_array.append(right_p.y)
                dist -= 1
            right_p = right_cl.interpolate(right_dist)
            x_array.append(right_p.x)
            y_array.append(right_p.y)

            axes.fill(x_array, y_array, color="#C0C0C0")
        else:
            if start_symbol:
                continue

        if lane_id == path[-1]:
            continue
        lane_obj = id_lane_dict[lane_id]
        left_array = lane_obj.left_way_array
        right_array = lane_obj.right_way_array
        # print(left_array[:, 0])
        x_array = []
        y_array = []
        for coord in left_array:
            x_array.append(coord[0])
            y_array.append(coord[1])
        for coord in reversed(right_array):
            x_array.append(coord[0])
            y_array.append(coord[1])
        axes.fill(x_array, y_array, color="#C0C0C0")
    
    last_lane_obj = id_lane_dict[path[-1]]
    point = Point(cl[-1])
    left_cl = LineString(last_lane_obj.left_way_array)
    right_cl = LineString(last_lane_obj.right_way_array)
    left_dist = left_cl.project(point)
    right_dist = right_cl.project(point)

    x_array = []
    y_array = []

    dist = 0
    while dist < left_dist:
        left_p = left_cl.interpolate(dist)
        x_array.append(left_p.x)
        y_array.append(left_p.y)
        dist += 1
    left_p = left_cl.interpolate(left_dist)
    x_array.append(left_p.x)
    y_array.append(left_p.y)

    dist = right_dist
    while dist > 0:
        right_p = right_cl.interpolate(dist)
        x_array.append(right_p.x)
        y_array.append(right_p.y)
        dist -= 1
    right_p = right_cl.interpolate(0)
    x_array.append(right_p.x)
    y_array.append(right_p.y)

    axes.fill(x_array, y_array, color="#C0C0C0")
    return axes