import numpy as np
from shapely.geometry import LineString, Point
from util_dir.geometry import is_on_line

class Lane(object):
    def __init__(self,
                 lane_id: int,
                 left_way_id: int,
                 right_way_id: int,
                 lane_type: str,
                 subtype: str,
                 region: str,
                 one_way: bool,
                 location: str):
        self.id = lane_id
        self.left_way_id = left_way_id
        self.right_way_id = right_way_id

        self.type = lane_type
        self.subtype = subtype
        self.region = region
        self.one_way = one_way  # if the way is one way direction
        self.location = location

        self.left_way_array = None
        self.right_way_array = None
        self.centerline_array = None

        self.left_way_type = None
        self.right_way_type = None

        self.lane_length = None

        self.speed_limit = []
        self.right_of_way = []
        self.all_way_stop = []
        self.traffic_sign = []

        self.yield_right = False
        self.yield_stop = False
        self.refers_list = []
        # The id of the lanes which have the right of way to this lanelet.
        self.right_of_way_list = []

        self.idx_dist_dict = {}

    def get_lane_length(self):
        if self.centerline_array is None:
            raise Exception("The center line of this lanelet is None!")
        else:
            cl_ls = LineString(coordinates=self.centerline_array)
            return cl_ls.length

    def get_speed_limit(self):
        if len(self.speed_limit) == 0:
            return None
        else:
            min_speed = 10e9
            for reg_obj in self.speed_limit:
                if reg_obj.speed_limit_num < min_speed:
                    min_speed = reg_obj.speed_limit_num

            return min_speed

    def get_point_length_to_start(self, target_p: np.ndarray) -> float:
        cl_ls = LineString(coordinates=self.centerline_array)
        target_point = Point(target_p)

        return cl_ls.project(target_point)

    def compute_dist_idx(self):
        dist = 0
        for i in range(self.centerline_array.shape[0]):
            if i == 0:
                self.idx_dist_dict[i] = 0
            else:
                dist += np.linalg.norm(self.centerline_array[i]-self.centerline_array[i-1])
                self.idx_dist_dict[i] = dist

    def get_idx(self, dist: float):
        if len(self.idx_dist_dict) == 0:
            self.compute_dist_idx()

        start = 0
        end = len(self.idx_dist_dict)

        while start <= end:
            mid = int(start + (end-start) / 2)
            if self.idx_dist_dict[mid] <= dist <= self.idx_dist_dict[mid+1]:
                return mid
            elif dist < self.idx_dist_dict[mid]:
                end = mid
            else:
                start = mid

    def get_dist_with_point(self, target_p: np.ndarray):
        dist = 0
        for i in range(self.centerline_array.shape[0]-1):
            if is_on_line(target_p, self.centerline_array[i], self.centerline_array[i+1]):
                dist += np.linalg.norm(target_p-self.centerline_array[i])
                return dist
            else:
                dist += np.linalg.norm(self.centerline_array[i+1]-self.centerline_array[i])
        raise Exception("Cannot find this point {}".format(target_p))