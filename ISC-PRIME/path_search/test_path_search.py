import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
import json
import matplotlib.pyplot as plt

from path_search.search_with_rule import path_search_rule as path_search_v1
from path_search.search_with_rule_v2 import path_search_rule as path_search_v2

from dataset.pandas_dataset import DatasetPandas, DATA_DICT
from hdmap.hd_map import HDMap


def get_target_vehicle(file_path: str):
    key_dict = {}
    with open(file_path, "r", encoding="UTF-8") as f:
        target_dict = json.load(f)

        for k in target_dict.keys():
            case_id = int(k)
            key_dict[case_id] = []
            for track_id in target_dict[k]:
                key_dict[case_id].append((case_id, track_id))
        # print(target_dict)
        f.close()
    return key_dict


if __name__ == '__main__':
    mode = "train"
    file_name = "DR_USA_Intersection_EP1"
    map_path = "/home/joe/Desktop/PredictionWithIRL/maps/" + file_name + ".osm"
    data_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "/" + file_name + "_" + mode + ".csv"
    target_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/" + mode + "_target_filter/" + file_name + ".json"

    dataset = DatasetPandas(data_path=data_path)
    hd_map = HDMap(osm_file_path=map_path)
    target_v = get_target_vehicle(file_path=target_path)

    for case_id in target_v.keys():
        case_data = dataset.get_case_data(case_id)
        for _, track_id in target_v[case_id]:
            # if (case_id, track_id) != (1, 15):
            #     continue
            print("{}-{}".format(case_id, track_id))
            track_full = dataset.get_track_data(case_id, track_id)
            future_xy = track_full[10:, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
            track_xy = track_full[:10, [DATA_DICT["x"], DATA_DICT["y"]]].astype("float")
            track_heading = track_full[:10, [DATA_DICT["psi_rad"]]].astype("float")
            cl_list, cor_path_list = path_search_v2(track_obs_xy=track_xy,
                                                    track_obs_yaw=track_heading,
                                                    case_data=case_data,
                                                    track_id=track_id,
                                                    hd_map=hd_map,
                                                    roundabout=False)

            cl_list_v1 = path_search_v1(
                track_obs_xy=track_xy,
                track_obs_heading=track_heading,
                case_data=case_data,
                track_id=track_id,
                hd_map=hd_map,
                roundabout=False
            )

            axes = plt.subplot(111)
            # axes = draw_lanelet_map(hd_map.lanelet_map, axes)
            for cl in cl_list:
                axes.plot(cl[:, 0], cl[:, 1], color="purple", linestyle="-")

            for cl in cl_list_v1:
                axes.plot(cl[:, 0], cl[:, 1], color="blue", linestyle="--", zorder=10)
            axes.plot(track_xy[:, 0], track_xy[:, 1], color="red")
            axes.scatter(track_xy[-1, 0], track_xy[-1, 1], color="red", marker="x", s=25)
            axes.plot(future_xy[:, 0], future_xy[:, 1], color="orange")
            axes.scatter(future_xy[-1, 0], future_xy[-1, 1], color="orange", marker="o", s=25)
            plt.show()