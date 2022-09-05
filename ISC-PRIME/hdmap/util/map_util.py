import numpy as np
import copy
from shapely.geometry import Point, Polygon
from hdmap.object.lane import Lane
from typing import List


def point_in_lanelet(lane: Lane, point: np.ndarray) -> bool:
    """
    Check if the point is in the lanelet or not.
    :param lane: The lanelet object.
    :param point: The coordinate of the point.
    :return: bool value
    """
    left_array = copy.deepcopy(lane.left_way_array)
    right_array = copy.deepcopy(lane.right_way_array)

    start_array = copy.deepcopy(left_array[0][:])
    start_array = start_array.reshape(-1, 2)

    bound_array = np.concatenate((left_array, np.flipud(right_array)), axis=0)
    bound_array = np.concatenate((bound_array, start_array), axis=0)

    lanelet = Polygon(bound_array)
    p = Point(point)

    if lanelet.intersection(p):
        return True
    else:
        return False


def point_in_lanelet_v2(lane: Lane, point: np.ndarray) -> bool:
    poly_x = []
    poly_y = []

    for idx in range(lane.left_way_array.shape[0]):
        poly_x.append(lane.left_way_array[idx][0])
        poly_y.append(lane.left_way_array[idx][1])

    for idx in range(lane.right_way_array.shape[0]-1, -1, -1):
        poly_x.append(lane.right_way_array[idx][0])
        poly_y.append(lane.right_way_array[idx][1])

    odd_node = False
    poly_sides = lane.left_way_array.shape[0] + lane.right_way_array.shape[0]
    j = poly_sides - 1
    x, y = point[0], point[1]

    for i in range(poly_sides):
        if (poly_y[i] < y and poly_y[j]) >= y or (poly_y[j] < y <= poly_y[i]) and (
                poly_x[i] <= x or poly_x[j] <= x):
            if poly_x[i] + (y - poly_y[i]) / (poly_y[j] - poly_y[i]) * (poly_x[j] - poly_x[i]) < x:
                odd_node = not odd_node

        j = i

    return odd_node


def find_all_poly_bboxes_overlapping_query_bbox(polygon_bboxes: np.ndarray, query_bbox: np.ndarray) -> np.ndarray:
    """
    Find all the overlapping polygon bounding boxes.
    Each bounding box has the following structure:
        bbox = np.array([x_min, y_min, x_max, y_max])

    :param polygon_bboxes: An array of shape (K,), each array element is a NumPy array of shape (4,) representing the
                           bounding box for a polygon or point cloud.
    :param query_bbox: An array of shape (4,) representing a 2d axis-aligned bounding box, with order
                       [min_x,min_y,max_x,max_y].
    :return: An integer array of shape (K, ) representing indices where overlap occurs.
    """
    query_min_x = query_bbox[0]
    query_min_y = query_bbox[1]
    query_max_x = query_bbox[2]
    query_max_y = query_bbox[3]

    bboxes_x1 = polygon_bboxes[:, 0]
    bboxes_x2 = polygon_bboxes[:, 2]

    bboxes_y1 = polygon_bboxes[:, 1]
    bboxes_y2 = polygon_bboxes[:, 3]

    # check if falls within range
    overlaps_left = (query_min_x <= bboxes_x2) & (bboxes_x2 <= query_max_x)
    overlaps_right = (query_min_x <= bboxes_x1) & (bboxes_x1 <= query_max_x)

    x_check1 = bboxes_x1 <= query_min_x
    x_check2 = query_min_x <= query_max_x
    x_check3 = query_max_x <= bboxes_x2
    x_subsumed = x_check1 & x_check2 & x_check3

    x_in_range = overlaps_left | overlaps_right | x_subsumed

    overlaps_below = (query_min_y <= bboxes_y2) & (bboxes_y2 <= query_max_y)
    overlaps_above = (query_min_y <= bboxes_y1) & (bboxes_y1 <= query_max_y)

    y_check1 = bboxes_y1 <= query_min_y
    y_check2 = query_min_y <= query_max_y
    y_check3 = query_max_y <= bboxes_y2
    y_subsumed = y_check1 & y_check2 & y_check3
    y_in_range = overlaps_below | overlaps_above | y_subsumed

    overlap_indxs = np.where(x_in_range & y_in_range)[0]
    return overlap_indxs


def get_lane_id_in_xy_bbox(query_x: float, query_y: float, hd_map, query_search_range_manhattan: float = 5.0) -> List[int]:
    """

    :param query_x: representing x coordinate of xy query location
    :param query_y: representation y coordinate of xy query location
    :param query_search_range_manhattan: search radius along axes
    :return: lane_ids: lane segment IDs that live within a bubble.
    """
    query_min_x = query_x - query_search_range_manhattan
    query_max_x = query_x + query_search_range_manhattan
    query_min_y = query_y - query_search_range_manhattan
    query_max_y = query_y + query_search_range_manhattan

    overlap_index = find_all_poly_bboxes_overlapping_query_bbox(
        polygon_bboxes=hd_map.halluc_bbox_table_array,
        query_bbox=np.array([query_min_x, query_min_y, query_max_x, query_max_y])
    )

    if len(overlap_index) == 0:
        return []

    neighborhood_lane_ids: List[int] = []

    for overlap_idx in overlap_index:
        lane_segment_id = hd_map.halluc_tableidx_to_laneid_dict[overlap_idx]
        neighborhood_lane_ids.append(lane_segment_id)

    return neighborhood_lane_ids


def get_polygon(lane_obj: Lane):
    reverse_right_array = np.flipud(lane_obj.right_way_array)

    start_array = copy.deepcopy(lane_obj.left_way_array[0][:])
    start_array = start_array.reshape(-1, 2)
    polygon_array = np.concatenate((lane_obj.left_way_array, reverse_right_array), axis=0)
    polygon_array = np.concatenate((polygon_array, start_array), axis=0)

    return polygon_array
