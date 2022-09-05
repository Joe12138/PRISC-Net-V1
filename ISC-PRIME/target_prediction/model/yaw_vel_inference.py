import os
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import numpy as np
import torch
import json

from target_prediction.dataloader.interaction_loader_v3 import InteractionInMem
from target_prediction.model.yaw_vel_predict import YawVelPredict
from util_dir.geometry import normalize_angle


class YawVelInference(object):
    def __init__(self, dataset_path: str, model_path: str, variable):
        self.variable = variable

        self.dataset_path = dataset_path
        self.dataset = InteractionInMem(root=dataset_path)
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

        self.model = YawVelPredict(
            in_channels=self.dataset.num_features,
            num_global_graph_layer=1,
            with_aux=True,
            device=self.device,
            variable=variable
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)

        if os.path.exists(os.path.join(self.dataset_path, "processed", "key.json")):
            f = open(os.path.join(self.dataset_path, "processed", "key.json"), "r", encoding="UTF-8")
            self.idx_dict = json.load(f)
            f.close()
        else:
            self.idx_dict = {}
            self.get_idx_dict()

        print(f"init_{variable}_complete")

    def get_idx_dict(self):
        seq_list = self.dataset.data.seq_id
        for idx, raw_file_name in enumerate(seq_list):
            # info_list = raw_file_name[:-4].split("_")
            # print(info_list)
            # scene_name = f"{info_list[1]}_{info_list[2]}_{info_list[3]}_{info_list[4]}"

            # case_id = int(info_list[5])
            # track_id = int(info_list[6])
            file_name = raw_file_name.values[0]

            self.idx_dict[file_name] = idx

        with open(os.path.join(self.dataset_path, "processed", "key.json"), "w", encoding="UTF-8") as f:
            json.dump(self.idx_dict, f)

    def inference(self, scene_name: str, case_id: int, track_id: int):
        key_str = f"{scene_name}_{case_id}_{track_id}"
        idx = self.idx_dict[key_str]

        data = self.dataset.get(idx)
        seq_id = data.seq_id.values[0]

        assert seq_id == f"{scene_name}_{case_id}_{track_id}", f"Wrong corresponding data {data.seq_id}"

        data.num_graphs = 1
        data.batch = None

        with torch.no_grad():
            if self.variable == "yaw":
                gt = data.yaw_array.unsqueeze(1).view(1, -1, 1).numpy()
            elif self.variable == "vx":
                gt = data.vx_array.unsqueeze(1).view(1, -1, 1).numpy()
            else:
                gt = data.vy_array.unsqueeze(1).view(1, -1, 1).numpy()

            out = self.model.inference(data.to(self.device))

            target_pred_se, offset_pred_se = out
            target_pred_se = target_pred_se.cpu().numpy()
            offset_pred_se = offset_pred_se.cpu().numpy()

            if self.variable == "yaw":
                real_pred = []
                for i in range(target_pred_se.shape[1]):
                    real_pred.append(normalize_angle(target_pred_se[0][i])+offset_pred_se[0][i])
            else:
                real_pred = [pred_y_k for pred_y_k in target_pred_se[0] + offset_pred_se[0]]

        return real_pred, gt


if __name__ == '__main__':
    dataset_path = "/home/joe/ServerBackup/final_version_rule_equal_interval_0_25/val_intermediate"
    model_path = "/home/joe/Desktop/Rule-PRIME/Model/yaw_output/06-27-18-51/best_YawVelPredict.pth"

    target_pred_inference = YawVelInference(dataset_path=dataset_path, model_path=model_path, variable="yaw")
    # target_pred_inference.get_idx_dict()

    # for key, value in target_pred_inference.idx_dict.items():
    #     print(key, " ", value)
    # target_pred_inference.idx_dict
    for key in target_pred_inference.idx_dict.keys():
        # if key != "DR_CHN_Roundabout_LN_1_1":
        #     continue
        key_list = key.split("_")
        scene_name = f"{key_list[0]}_{key_list[1]}_{key_list[2]}_{key_list[3]}"
        result, gt = target_pred_inference.inference(scene_name=scene_name, case_id=key_list[4], track_id=key_list[5])
        print(result)
        print(gt)
        break