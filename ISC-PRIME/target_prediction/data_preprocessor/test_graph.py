import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point

import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")


def get_lane_vector(lane_coord: np.ndarray, interval: float = 1.0):
    ls = LineString(coordinates=lane_coord)
    dist_list = np.arange(0, ls.length, 1)

    coord_list = []

    for i in dist_list:
        point = ls.interpolate(i)
        coord_list.append([point.x, point.y])

    return np.asarray(coord_list)


if __name__ == '__main__':
    data_path = "/home/joe/ServerBackup/final_version_rule/val_intermediate/raw"
    # data_path = "/home/joe/Dataset/target_data/val_intermediate/raw"
    file_list = os.listdir(data_path)
    for file in file_list:
        if file != "features_DR_CHN_Roundabout_LN_1_1.pkl":
            continue
        raw_path = f"{data_path}/{file}"

        raw_data = pd.read_pickle(raw_path)
        graph = raw_data["graph"].values

        ctrs = graph[0]["ctrs"]
        lane_idcs = graph[0]["lane_idcs"]

        axes = plt.subplot(111)

        for lid in np.unique(lane_idcs):
            [indics] = np.where(lane_idcs == lid)
            cur_lane = ctrs[indics]
            lane_array = get_lane_vector(cur_lane)
            # print(indics)

            for i in range(lane_array.shape[0]-1):
                axes.annotate("", xy=lane_array[i], xytext=lane_array[i+1], arrowprops=dict(arrowstyle="->"))
        # print(ctrs)
        # print("hello")
        plt.xlim((-50, 50))
        plt.ylim((-50, 50))
        plt.show()
        plt.cla()
        plt.clf()