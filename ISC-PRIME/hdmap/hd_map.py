import sys

from joblib import wrap_non_picklable_objects, delayed

sys.path.append("/home/joe/Desktop/TRCVTPP/RulePRIMEV2/")
import lanelet2
import numpy as np
from lanelet2_parser import Lanelet2Parser
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.axes
from typing import Dict, Tuple, Any, List
from shapely.geometry import LineString

from hdmap.object.node import Node
from hdmap.object.way import Way
from hdmap.object.lane import Lane
from hdmap.object.graph_node import GraphNode
from hdmap.object.regulatory_element import TrafficSign, SpeedLimit, AllWayStop, RightOfWay
from hdmap.util.map_util import point_in_lanelet_v2
from util_dir.geometry import get_angle, normalize_angle


class HDMap(object):
    def __init__(self, osm_file_path: str, lat_origin: float = 0.0, lon_origin: float = 0.0):
        self.projector = lanelet2.projection.UtmProjector(lanelet2.io.Origin(lat_origin, lon_origin))
        self.lanelet_map = lanelet2.io.load(osm_file_path, self.projector)

        lanelet2_parser = Lanelet2Parser()
        self.osm_data = lanelet2_parser.parse(osm_file_path)

        self.halluc_bbox_table: List[List[int]] = []
        self.halluc_tableidx_to_laneid_dict = {}

        self.id_reg_dict = self.parser_regulatory_element(reg_element_list=self.get_regulatory_element_list())
        self.id_node_dict = self.get_node()
        self.id_way_dict = self.get_way()
        self.id_lane_dict, self.id_lanelet_dict = self.get_lane()

        self.halluc_bbox_table_array = np.asarray(self.halluc_bbox_table)

        self.traffic_rule = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                          lanelet2.traffic_rules.Participants.Vehicle)
        self.graph = lanelet2.routing.RoutingGraph(self.lanelet_map, self.traffic_rule)

        self.road_graph, self.directed_graph, self.undirected_graph = self.construct_graph()

    def get_regulatory_element_list(self) -> list:
        """
        Get regulatory element list with Lanelet2Parser.
        :return: A list of regulatory element which is [(relation_id, relation_obj), ...]
        """
        reg_element_list = []
        for relation_id, relation_obj in self.osm_data.relations.items():
            for ele in relation_obj.tags:
                if ele.key == "type":
                    if ele.value == "regulatory_element":
                        reg_element_list.append((relation_id, relation_obj))
                else:
                    continue
        return reg_element_list

    @staticmethod
    def parser_regulatory_element(reg_element_list: list) -> dict:
        """
        Parser the regulatory element with Lanelet2Parser.
        :param reg_element_list: The regulatory element list which is [(reg_id, reg_obj), ...]
        :return: The dictionary of regulatory element.
        """
        id_reg_dict = {}

        for reg_ele in reg_element_list:
            reg_id = int(reg_ele[0])
            reg_obj = reg_ele[1]

            tag_dict = {}
            for tag_ele in reg_obj.tags:
                tag_dict[tag_ele.key] = tag_ele.value

            if tag_dict["subtype"] == "speed_limit":
                speed_limit_obj = SpeedLimit(reg_id=reg_id,
                                             reg_type=tag_dict["type"],
                                             subtype=tag_dict["subtype"],
                                             sign_type=tag_dict["sign_type"])
                id_reg_dict[reg_id] = speed_limit_obj
            elif tag_dict["subtype"] == "right_of_way":
                member_dict = {"refers": [], "ref_line": [], "right_of_way": [], "yield": []}
                for member in reg_obj.members:
                    member_dict[member.role].append((int(member.ref), member.type))
                right_of_way_obj = RightOfWay(reg_id=reg_id,
                                                reg_type=tag_dict["type"],
                                                subtype=tag_dict["subtype"],
                                                refers=member_dict["refers"],
                                                ref_line=member_dict["ref_line"],
                                                right_of_way=member_dict["right_of_way"],
                                                yield_list=member_dict["yield"])

                id_reg_dict[reg_id] = right_of_way_obj
            elif tag_dict["subtype"] == "all_way_stop":
                member_dict = {"refers": [], "ref_line": [], "yield": []}
                for member in reg_obj.members:
                    member_dict[member.role].append((int(member.ref), member.type))
                all_way_stop_obj = AllWayStop(reg_id=reg_id,
                                              reg_type=tag_dict["type"],
                                              subtype=tag_dict["subtype"],
                                              refers=member_dict["refers"],
                                              ref_line=member_dict["ref_line"],
                                              yield_list=member_dict["yield"])
                id_reg_dict[reg_id] = all_way_stop_obj
            else:
                raise Exception("No this regulatory element subtype: {}".format(tag_dict["subtype"]))

        return id_reg_dict

    def get_node(self) -> Dict[int, Node]:
        id_node_dict = {}

        for node in self.lanelet_map.pointLayer:
            node_obj = Node(node_id=int(node.id), x=float(node.x), y=float(node.y))

            id_node_dict[int(node.id)] = node_obj

        return id_node_dict

    def get_way(self) -> Dict[int, Way]:
        id_way_dict = {}

        for way in self.lanelet_map.lineStringLayer:
            ref_node_list = []

            way_type = way.attributes["type"]
            try:
                subtype = way.attributes["subtype"]
            except KeyError:
                subtype = None

            for ref_p in way:
                ref_node_list.append(int(ref_p.id))
                self.id_node_dict[int(ref_p.id)].way_list.append(int(way.id))

            way_obj = Way(way_id=int(way.id),
                          way_type=way_type,
                          way_subtype=subtype,
                          ref_node_list=ref_node_list)
            id_way_dict[int(way.id)] = way_obj

            if way_type == "traffic_sign":
                traffic_sign = TrafficSign(reg_id=int(way.id),
                                           reg_type=way_type,
                                           subtype=subtype,
                                           ref_node_list=ref_node_list)
                traffic_sign.sign_meaning = traffic_sign.get_sign_meaning()
                self.id_reg_dict[int(way.id)] = traffic_sign
        return id_way_dict

    @staticmethod
    def get_way_array(way) -> np.ndarray:
        """
        Get way list.
        :param way:
        :return: A list of the coordinate of the way points.
        """
        way_list = []
        for element in way:
            way_list.append((float(element.x), float(element.y)))

        return np.asarray(way_list)

    def get_lane(self) -> Tuple[Dict[int, Lane], Dict[int, Any]]:
        id_lane_dict = {}
        id_lanelet_dict = {}

        for lane in self.lanelet_map.laneletLayer:
            lane_id = int(lane.id)
            id_lanelet_dict[lane_id] = lane

            location = lane.attributes["location"]
            if lane.attributes["one_way"] == "yes" or lane.attributes["one_way"] == "y":
                one_way = True
            elif lane.attributes["one_way"] == "no" or lane.attributes["one_way"] == "n":
                one_way = False
            else:
                raise Exception("No this one_way: {}".format(lane.attributes["one_way"]))

            try:
                region = lane.attributes["region"]
            except KeyError:
                region = None
            subtype = lane.attributes["subtype"]
            lane_type = lane.attributes["type"]

            centerline_array = self.get_way_array(way=lane.centerline)
            left_way_array = self.get_way_array(way=lane.leftBound)
            right_way_array = self.get_way_array(way=lane.rightBound)

            lane_obj = Lane(lane_id=lane_id,
                            left_way_id=int(lane.leftBound.id),
                            right_way_id=int(lane.rightBound.id),
                            lane_type=lane_type,
                            subtype=subtype,
                            region=region,
                            one_way=one_way,
                            location=location)

            lane_obj.left_way_array = left_way_array
            lane_obj.right_way_array = right_way_array
            lane_obj.centerline_array = centerline_array

            min_x = min(np.min(left_way_array[:, 0]), np.min(right_way_array[:, 0]))
            max_x = max(np.max(left_way_array[:, 0]), np.max(right_way_array[:, 0]))
            min_y = min(np.min(left_way_array[:, 1]), np.min(right_way_array[:, 1]))
            max_y = max(np.max(left_way_array[:, 1]), np.max(right_way_array[:, 1]))
            self.halluc_bbox_table.append([min_x, min_y, max_x, max_y])

            self.halluc_tableidx_to_laneid_dict[len(self.halluc_bbox_table)-1] = lane_id

            lane_obj.left_way_type = (self.id_way_dict[lane_obj.left_way_id].type,
                                      self.id_way_dict[lane_obj.left_way_id].subtype)
            lane_obj.right_way_type = (self.id_way_dict[lane_obj.right_way_id].type,
                                       self.id_way_dict[lane_obj.right_way_id].subtype)

            lane_obj.lane_length = lane_obj.get_lane_length()

            reg_element = lane.regulatoryElements
            for reg_ele in reg_element:
                reg_obj = self.id_reg_dict[int(reg_ele.id)]
                if reg_obj.subtype == "speed_limit":
                    lane_obj.speed_limit.append(reg_obj)
                elif reg_obj.subtype == "right_of_way":
                    lane_obj.right_of_way.append(reg_obj)

                    yield_set = set()
                    for yield_lane_id, relation in reg_obj.yield_list:
                        yield_set.add(yield_lane_id)

                    if lane_obj.id in yield_set:
                        lane_obj.yield_right = True

                        for right_way_id, relation in reg_obj.right_of_way:
                            lane_obj.right_of_way_list.append(right_way_id)

                        for refer_id, relation in reg_obj.refers:
                            try:
                                if self.id_reg_dict[refer_id].type == "traffic_sign":
                                    lane_obj.traffic_sign.append(refer_id)
                            except KeyError:
                                pass
                            lane_obj.refers_list.append(refer_id)
                elif reg_obj.subtype == "all_way_stop":
                    lane_obj.all_way_stop.append(reg_obj)
                    yield_set = set()

                    for yield_lane_id, relation in reg_obj.yield_list:
                        yield_set.add(yield_lane_id)

                    if lane_obj.id in yield_set:
                        lane_obj.yield_stop = True

                        for refer_id, relation in reg_obj.refers:
                            if self.id_reg_dict[refer_id].type == "traffic_sign":
                                lane_obj.traffic_sign.append(refer_id)
                            lane_obj.refers_list.append(refer_id)
                else:
                    raise Exception("No this reg subtype: {}".format(reg_obj.subtype))
            id_lane_dict[lane_id] = lane_obj
        return id_lane_dict, id_lanelet_dict

    def find_lanelet(self, pos: np.ndarray):
        lanelet_list = []
        for lane_id, lane_obj in self.id_lane_dict.items():
            if point_in_lanelet_v2(lane_obj, pos):
                lanelet_list.append(lane_id)

        return lanelet_list

    def construct_graph(self) -> Tuple[nx.DiGraph, nx.DiGraph, nx.DiGraph]:
        road_graph = nx.DiGraph()
        directed_graph = nx.DiGraph()
        undirected_graph = nx.DiGraph()

        for lane_id, lane_obj in self.id_lanelet_dict.items():
            direct_res = set(self.graph.following(lane_obj, withLaneChanges=False))
            lane_change_res = set(self.graph.following(lane_obj, withLaneChanges=True))
            lane_change_res = lane_change_res-direct_res

            for node in direct_res:
                node_id = int(node.id)
                road_graph.add_edge(lane_id, node_id, lane_change=False)
                directed_graph.add_edge(lane_id, node_id)

            for node in lane_change_res:
                node_id = int(node.id)
                road_graph.add_edge(lane_id, node_id, lane_change=True)
                undirected_graph.add_edge(lane_id, node_id)

        return road_graph, directed_graph, undirected_graph

    def get_lane_info(self, lane_id: int):
        lane_obj = self.id_lane_dict[lane_id]

        has_control, turn_direction, is_intersect = False, None, False

        if len(lane_obj.traffic_sign) > 0:
            has_control = True

        if len(lane_obj.right_of_way_list) > 0 or lane_obj.yield_right or lane_obj.yield_stop:
            is_intersect = True

        sl = lane_obj.get_speed_limit()
        speed_limit = 100/3.6 if sl is None else sl

        start_array = lane_obj.centerline_array[1]-lane_obj.centerline_array[0]
        end_array = lane_obj.centerline_array[-1]-lane_obj.centerline_array[-2]
        angle_diff = get_angle(start_array, end_array)

        if abs(angle_diff) < np.pi/4:
            return has_control, turn_direction, is_intersect, speed_limit
        else:
            x_vec = np.array([1, 0])
            start_angle = get_angle(x_vec, start_array)
            end_angle = get_angle(x_vec, end_array)

            nor_angle = normalize_angle(end_angle-start_angle)
            if nor_angle > 0:
                turn_direction = "LEFT"
            else:
                turn_direction = "RIGHT"

            return has_control, turn_direction, is_intersect, speed_limit


if __name__ == '__main__':
    from hdmap.visual.map_vis import draw_lanelet_map, plot_lane_id, plot_road_network
    from path_search.visual.path_viz import plot_path
    from path_search.search_with_rule import find_all_paths, find_directed_paths
    map_file_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps/DR_USA_Intersection_EP1.osm"
    hd_map = HDMap(osm_file_path=map_file_path)

    road_g, directed_g, undirected_g = hd_map.construct_graph()

    axes = plt.subplot(111)
    axes = draw_lanelet_map(hd_map.lanelet_map, axes)
    # axes = plot_road_network(road_g, hd_map.id_lane_dict, axes)
    # axes = plot_lane_id(axes, hd_map.id_lane_dict)
    plt.show()

    # leaves = [v for v, d in directed_g.out_degree() if d == 0]
    # print(leaves)
    # all_path = nx.all_simple_paths(road_g, 30027, leaves)
    # for path in all_path:
    #     print(path)
    #     axes = plt.subplot(111)
    #     axes = plot_road_network(road_graph=road_g, id_lane_dict=hd_map.id_lane_dict, axes=axes)
    #     axes = plot_path(path=path, id_lane_dict=hd_map.id_lane_dict, axes=axes)
    #     plt.show()


    # Test path search
    # path_list = find_all_paths(lane_list=[30027], hd_map=hd_map)
    # # path_list = find_directed_paths(lane_list=[30048], hd_map=hd_map)
    # all_path = path_list[0]
    # for path in all_path:
    #     axes = plt.subplot(111)
    #     axes = plot_road_network(road_graph=road_g, id_lane_dict=hd_map.id_lane_dict, axes=axes)
    #     axes = plot_path(path=path, id_lane_dict=hd_map.id_lane_dict, axes=axes)
    #     plt.show()
    #     print(path)

