import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import os
import json

import pandas as pd

DATA_DICT = {
    "case_id": 0,
    "track_id": 1,
    "frame_id": 2,
    "timestamp_ms": 3,
    "agent_type": 4,
    "x": 5,
    "y": 6,
    "vx": 7,
    "vy": 8,
    "psi_rad": 9,
    "length": 10,
    "width": 11,
    "track_to_predict": 12
}

test_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/test_single-agent"

save_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/test_target_filter"

file_list = os.listdir(test_path)

for file_name in file_list:
    data_path = os.path.join(test_path, file_name)

    data_df = pd.read_csv(data_path)

    data_df_filter = data_df[data_df["track_to_predict"] == 1]
    case_set = data_df_filter[["case_id", "track_id"]].drop_duplicates(["case_id", "track_id"]).values.astype(int)
    # print("hello")

    target_veh_dict = {}
    for i in range(case_set.shape[0]):
        if case_set[i][0] not in target_veh_dict.keys():
            target_veh_dict[str(case_set[i][0])] = [int(case_set[i][1])]
        else:
            target_veh_dict[str(case_set[i][0])].append(int(case_set[i][1]))

    save_path_a = os.path.join(save_path, file_name[:-8] + ".json")
    json_obj = json.dumps(target_veh_dict)
    with open(save_path_a, "w") as f:
        f.write(json_obj)
        f.close()
