import os.path
from re import S
import sys
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")

import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from target_prediction.dataloader.interaction_loader_v3 import InteractionInMem
# from target_prediction.model.target_predict_more_feature import TargetPredict
from target_prediction.model.target_predict import TargetPredict
from hdmap.hd_map import HDMap
from hdmap.visual.map_vis import draw_lanelet_map


def convert_coord(traj, orig, rot):
    traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
    return traj_converted


class TargetPredInference(object):
    def __init__(self, dataset_path: str, model_path: str):
        self.dataset_path = dataset_path
        self.dataset = InteractionInMem(root=dataset_path)
        self.device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

        self.model = TargetPredict(
            in_channels=self.dataset.num_features,
            num_global_graph_layer=1,
            with_aux=True,
            device=self.device
        )
        if "ckpt" in model_path:
            m_mode = "c"
        else:
            m_mode = "m"
        self.load(model_path, mode=m_mode)
        # self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model = self.model.to(self.device)

        if os.path.exists(os.path.join(self.dataset_path, "processed", "key.json")):
            f = open(os.path.join(self.dataset_path, "processed", "key.json"), "r", encoding="UTF-8")
            self.idx_dict = json.load(f)
            f.close()
        else:
            self.idx_dict = {}
            self.get_idx_dict()

        print("init_complete")

    def load(self, load_path, mode='c'):
        """
        loading function to load the ckpt or model
        :param mode: str, "c" for checkpoint, or "m" for model
        :param load_path: str, the path of the file to be load
        :return:
        """
        if mode == 'c':
            # load ckpt
            ckpt = torch.load(load_path, map_location=self.device)
            try:
                self.model.load_state_dict(ckpt["model_state_dict"])
            except:
                raise Exception("[Trainer]: Error in loading the checkpoint file {}".format(load_path))
        elif mode == 'm':
            try:
                self.model.load_state_dict(torch.load(load_path, map_location=self.device))
            except:
                raise Exception("[Trainer]: Error in loading the model file {}".format(load_path))
        else:
            raise NotImplementedError

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

        # plot tool
        # plot = False
        # map_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps"
        # hd_map = HDMap(osm_file_path=os.path.join(map_path, f"{scene_name}.osm"))

        with torch.no_grad():
            origs = data.orig.numpy()
            rots = data.rot.numpy()

            gt = data.y.unsqueeze(1).view(1, -1, 2).cumsum(axis=1).numpy()

            out = self.model.inference(data.to(self.device))

            target_pred_se, offset_pred_se = out
            target_pred_se = target_pred_se.cpu().numpy()
            offset_pred_se = offset_pred_se.cpu().numpy()

            real_coord_pred = [convert_coord(pred_y_k, origs[0], rots[0]) for pred_y_k in target_pred_se[0]+offset_pred_se[0]]
            gt_trajectories = convert_coord(gt[0], origs[0], rots[0])
            # print(gt_trajectories)
            min_fde = 10e9
            max_fde = -min_fde
            avg_fde = 0
            for i in range(6):
                diff = np.linalg.norm(real_coord_pred[i]-gt_trajectories[-1])

                if diff < min_fde:
                    min_fde = diff
                if diff > max_fde:
                    max_fde = diff

                avg_fde += diff
            
            avg_fde /= 6

            # if plot:
            #     axes = plt.subplot(111)
            #     axes = draw_lanelet_map(laneletmap=hd_map.lanelet_map, axes=axes)
            #     axes.plot(gt_trajectories[:, 0], gt_trajectories[:, 1], color="purple")
            #     axes.scatter(gt_trajectories[-1, 0], gt_trajectories[-1, 1], color="purple", s=25)
            #     axes.set_title(f"min={round(min_fde, 3)}, max={round(max_fde, 3)}, avg={round(avg_fde, 3)}")
            #     for i in range(6):
            #         axes.scatter(real_coord_pred[i][0][0], real_coord_pred[i][0][1], color="green", s=25, zorder=10)

            #     plt.savefig(f"/home/joe/Desktop/TRCVTPP/RulePRIMEV2/target_prediction/visual_res/{scene_name}_{case_id}_{track_id}.png", dpi=600)
            #     plt.cla()
            #     plt.clf()
        return real_coord_pred


if __name__ == '__main__':
    dataset_path = "/home/joe/ServerBackup/final_version_rule/val_intermediate"
    model_path = "/home/joe/Desktop/target_pred_mode/new_target_pred/best_TargetPredict.pth"

    target_pred_inference = TargetPredInference(dataset_path=dataset_path, model_path=model_path)
    # target_pred_inference.get_idx_dict()

    # for key, value in target_pred_inference.idx_dict.items():
    #     print(key, " ", value)
    # target_pred_inference.idx_dict
    for key in target_pred_inference.idx_dict.keys():
        if key != "DR_CHN_Roundabout_LN_1_1":
            continue
        key_list = key.split("_")
        scene_name = f"{key_list[0]}_{key_list[1]}_{key_list[2]}_{key_list[3]}"
        result = target_pred_inference.inference(scene_name=scene_name, case_id=key_list[4], track_id=key_list[5])
    print(result)
