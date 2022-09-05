import matplotlib
import matplotlib.axes
import matplotlib.pyplot as plt

from typing import Dict, List
from shapely.geometry import LineString
from hdmap.object.lane import Lane
from hdmap.util.map_util import get_polygon

import networkx

import random


def set_visible_area(laneletmap, axes):
    min_x = 10e9
    min_y = 10e9
    max_x = -10e9
    max_y = -10e9

    for point in laneletmap.pointLayer:
        min_x = min(point.x, min_x)
        min_y = min(point.y, min_y)
        max_x = max(point.x, max_x)
        max_y = max(point.y, max_y)

    axes.set_aspect('equal', adjustable='box')
    axes.set_xlim([min_x - 5, max_x + 5])
    axes.set_ylim([min_y - 5, max_y + 5])


def draw_lanelet_map(laneletmap, axes):
    assert isinstance(axes, matplotlib.axes.Axes)

    # set_visible_area(laneletmap, axes)
    unknown_linestring_types = list()

    for ls in laneletmap.lineStringLayer:

        if "type" not in ls.attributes.keys():
            raise RuntimeError("ID " + str(ls.id) + ": Linestring type must be specified")
        elif ls.attributes["type"] == "curbstone":
            type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "line_thin":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="black", linewidth=0.5, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "line_thick":
            if "subtype" in ls.attributes.keys() and ls.attributes["subtype"] == "dashed":
                type_dict = dict(color="black", linewidth=0.5, zorder=10, dashes=[10, 10])
            else:
                type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "pedestrian_marking":
            type_dict = dict(color="black", linewidth=0.5, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "bike_marking":
            type_dict = dict(color="black", linewidth=0.5, zorder=10, dashes=[5, 10])
        elif ls.attributes["type"] == "stop_line":
            type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "virtual":
            type_dict = dict(color="black", linewidth=0.5, zorder=10, dashes=[2, 5])
        elif ls.attributes["type"] == "road_border":
            type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "guard_rail":
            type_dict = dict(color="black", linewidth=0.5, zorder=10)
        elif ls.attributes["type"] == "traffic_sign":
            continue
        elif ls.attributes["type"] == "building":
            type_dict = dict(color="pink", zorder=1, linewidth=5)
        elif ls.attributes["type"] == "spawnline":
            if ls.attributes["spawn_type"] == "start":
                type_dict = dict(color="green", zorder=11, linewidth=0.5)
            elif ls.attributes["spawn_type"] == "end":
                type_dict = dict(color="red", zorder=11, linewidth=0.5)

        else:
            if ls.attributes["type"] not in unknown_linestring_types:
                unknown_linestring_types.append(ls.attributes["type"])
            continue

        ls_points_x = [pt.x for pt in ls]
        ls_points_y = [pt.y for pt in ls]
        plt.plot(ls_points_x, ls_points_y, **type_dict)

    if len(unknown_linestring_types) != 0:
        print("Found the following unknown types, did not plot them: " + str(unknown_linestring_types))

    return axes


def plot_lane_id(axes: matplotlib.axes, id_lane_dict: Dict[int, Lane]) -> matplotlib.axes:

    for lane_id, lane_obj in id_lane_dict.items():
        cl_ls = LineString(coordinates=lane_obj.centerline_array)
        center_p = cl_ls.interpolate(cl_ls.length/10)

        axes.text(center_p.x, center_p.y, str(lane_id), color="purple")

    return axes


def plot_road_network(road_graph: networkx.DiGraph, id_lane_dict: Dict[int, Lane], axes: matplotlib.axes):
    id_set = set()
    for s_id, e_id, change_dict in road_graph.edges.data():
        s_lane_obj = id_lane_dict[s_id]
        s_lane_ls = LineString(coordinates=s_lane_obj.centerline_array)
        s_p = s_lane_ls.interpolate(s_lane_ls.length/2)

        if s_id not in id_set:
            axes.text(s_p.x, s_p.y, str(s_id), color="purple")
            id_set.add(s_id)
        axes.scatter(s_p.x, s_p.y, marker="o", color="black", s=25)

        e_lane_obj = id_lane_dict[e_id]
        e_lane_ls = LineString(coordinates=e_lane_obj.centerline_array)
        e_p = e_lane_ls.interpolate(e_lane_ls.length / 2)

        if e_id not in id_set:
            axes.text(e_p.x, e_p.y, str(e_id), color="purple")
            id_set.add(e_id)
        axes.scatter(e_p.x, e_p.y, marker="o", color="black", s=25)

        if change_dict["lane_change"]:
            axes.annotate("", xy=(e_p.x, e_p.y), xytext=(s_p.x, s_p.y),
                          arrowprops=dict(arrowstyle="->", color="red", linestyle="--"), alpha=0.7)
        else:
            axes.annotate("", xy=(e_p.x, e_p.y), xytext=(s_p.x, s_p.y),
                          arrowprops=dict(arrowstyle="->", color="black", linestyle="--"), alpha=0.7)

    return axes


def plot_lane_list(id_list: List, id_lane_dict: Dict[int, Lane], axes: matplotlib.axes):
    for id in id_list:
        lane_obj = id_lane_dict[id]
        polygon = get_polygon(lane_obj=lane_obj)

        axes.fill(polygon[:, 0], polygon[:, 1], color="green")

    return axes
