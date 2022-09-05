import os
import sys
import json
import logging
sys.path.append("/home/joe/Desktop/Rule-PRIME/RulePRIME")
from prime_evaluator.main import main

import torch
import torchvision
from torch.utils.data import DataLoader

import numpy as np

from prime_evaluator.data_type import get_datatype_by_name
from models import get_model_by_name
from losses import get_loss_by_name

from tqdm import tqdm

from prime_evaluator.utils.calcu_utils import calcu_min_ade_fde

from prime_evaluator.utils.parsing import parse_arguments

from prime_evaluator.utils.eval_forecasting import (
    get_accuracy_statistic,
    get_fde_statistic,
    compute_forecasting_metrics
)

from utils.config import (
    _MAX_GUESSES_NUM, _MISSING_THRESHOLD, _HISTOGRAM_BINS
)

from prime_evaluator.data_type.centerline_xy_dataset import trans_back_sample_batched

from DenseTNT.src.modeling.vectornet import VectorNet
from DenseTNT.src import utils, structs
import argparse

from hdmap.hd_map import HDMap
from hdmap.visual.map_vis import draw_lanelet_map

import matplotlib.pyplot as plt

import copy

path_prefix = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2"
mode = "val"
data_path = os.path.join(path_prefix, mode)
map_path = os.path.join(path_prefix, "maps")
target_veh_path = os.path.join(path_prefix, f"{mode}_target_filter")

class ComparePredictor():
    def __init__(
        self, args, 
        dense_tnt_args,
        prime_path, 
        rule_prime_path, 
        prime_model_path,
        rule_model_path,
        target_veh_path,
        map_path
        ) -> None:
        self.args = args
        self.save_dir = os.path.join("/home/joe/Desktop/trained_model", "best_prime_rule_prime_dense_tnt_comparison_5")
        os.makedirs(self.save_dir, exist_ok=True)
        self.use_cuda = args.use_cuda
        self.net_name = args.net_name

        self.prime_dataset_val, self.prime_dataloader_val = self.create_dataloader(
            args, prime_path
        )
        self.rule_prime_dataset_val, self.rule_prime_dataloader_val = self.create_dataloader(
            args, rule_prime_path
        )

        self.prime_model = self.create_model(args)
        self.rule_prime_model = self.create_model(args)

        self.target_veh_list = self.get_all_target_veh(target_veh_path)

        self.map_dict = self.get_map_dict(map_path)


        self.init_predictor(args, prime_model_path, rule_model_path)
        
        device = torch.device("cuda:0")
        print("Loading Evalute Dataset", dense_tnt_args.data_dir)
        if dense_tnt_args.argoverse:
            from DenseTNT.src.dataset_interaction import Dataset
        save_dir = "/home/joe/Desktop/trained_model/dense_tnt/val"
        self.dense_tnt_eval_dataset = Dataset(
            args=dense_tnt_args,
            batch_size=dense_tnt_args.train_batch_size,
            data_path=data_path,
            map_path=map_path,
            target_veh_path=target_veh_path,
            mode=mode,
            save_dir=save_dir
        )
        self.model = VectorNet(dense_tnt_args)
        if dense_tnt_args.model_recover_path is None:
            raise ValueError("model_recover_path not specified.")

        model_recover = torch.load(dense_tnt_args.model_recover_path)
        self.model.load_state_dict(model_recover)
        
        if 'set_predict-train_recover' in dense_tnt_args.other_params and 'complete_traj' in dense_tnt_args.other_params:
            model_recover = torch.load(dense_tnt_args.other_params['set_predict-train_recover'])
        utils.load_model(self.model.decoder.complete_traj_cross_attention, model_recover, prefix='decoder.complete_traj_cross_attention.')
        utils.load_model(self.model.decoder.complete_traj_decoder, model_recover, prefix='decoder.complete_traj_decoder.')

        self.model.to(device)
        self.model.eval()
        print("load model successfully!")
    
    @staticmethod
    def get_map_dict(map_dir: str):
        map_dict = {}

        map_file_list = os.listdir(map_dir)

        for file in map_file_list:
            if "xy" in file:
                continue
            print(file)
            scene_name = file[:-4]
            hd_map = HDMap(osm_file_path=os.path.join(map_dir, file))

            map_dict[scene_name] = hd_map

        return map_dict

    
    def get_target_veh_list(self, target_veh_path: str, file_name: str):
        target_veh_list = []
        scene_name = file_name[:-5]
        with open(os.path.join(target_veh_path, file_name), "r", encoding="UTF-8") as f:
            target_dict = json.load(f)

            for k in target_dict.keys():
                case_id = int(k)

                for track_id in target_dict[k]:
                    target_veh_list.append((scene_name, case_id, track_id))

            f.close()
        return target_veh_list

    def get_all_target_veh(self, target_veh_path: str):
        file_list = os.listdir(target_veh_path)

        target_veh_list = []
        for file_name in file_list:
            target_veh = self.get_target_veh_list(target_veh_path, file_name)

            target_veh_list.extend(target_veh)

            # break
        
        return target_veh_list

    def create_dataloader(self, args, data_path):
        dataset_val = get_datatype_by_name(args.flag_dataset, data_path, args, augment=False)
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=dataset_val.collate_fn,
            pin_memory=True
        )

        return dataset_val, dataloader_val

    def create_model(self, args):
        model = get_model_by_name(args.net_name, args)

        if args.use_cuda:
            model = model.cuda()
        return model

    def init_predictor(self, args, prime_model_path, rule_model_path):
        if args.mode in ["val"]:
            if os.path.exists(prime_model_path):
                state_dict = torch.load(prime_model_path)['state_dict']
                self.prime_model.load_state_dict(state_dict)
            else:
                assert False, f"No pretrained weights could be loaded from {prime_model_path}"
            
            if os.path.exists(rule_model_path):
                state_dict = torch.load(rule_model_path)['state_dict']
                self.rule_prime_model.load_state_dict(state_dict)
            else:
                assert False, f"No pretrained weights could be loaded from {rule_model_path}"
    
    def validate(self):
        file_prefix = "seq"
        device = torch.device("cuda:0")
        with torch.no_grad():
            for scene_name, case_id, track_id in tqdm(self.target_veh_list):
                # print("scene_name:{}; case_id:{}; track_id:{}".format(scene_name, case_id, track_id))
                hd_map = self.map_dict[scene_name]
                seq_id = f"{file_prefix}_{scene_name}_{case_id}_{track_id}.pkl"
                try:
                    prime_idx = self.prime_dataset_val.seq_ids.index(seq_id)
                    
                    rule_prime_idx = self.rule_prime_dataset_val.seq_ids.index(seq_id)
                    prime_data = self.prime_dataset_val.__getitem__(prime_idx)
                    rule_data = self.rule_prime_dataset_val.__getitem__(rule_prime_idx)

                    prime_batch = [prime_data]
                    rule_batch = [rule_data]
                    
                    dense_instance = self.dense_tnt_eval_dataset.get_item(scene_name, case_id, track_id)
                    dense_batch = [dense_instance]

                    _, prime_input = self.prime_dataset_val.collate_fn(prime_batch)
                    _, rule_input = self.rule_prime_dataset_val.collate_fn(rule_batch)
                    dense_input = utils.batch_list_to_batch_tensors(dense_batch)
                except Exception as e:
                    # print(e)
                    continue
                
                try:
                    prime_output = self.prime_model(prime_input)
                    rule_output = self.rule_prime_model(rule_input)
                    dense_output, pred_score, _ = self.model(dense_input, device)
                except Exception as e:
                    continue
                # print(dense_output)

                prime_filter_output, prime_filter_prbs, prime_nofilter_output, prime_sample_batched = \
                    self.prime_model.post_process(self.prime_dataset_val, prime_output, prime_input, phase='validate')
                prime_metrics = get_accuracy_statistic(self.prime_dataset_val, self.args.pred_len, prime_nofilter_output, prime_sample_batched)
                prime_metrics_flt = get_accuracy_statistic(self.prime_dataset_val, self.args.pred_len, prime_filter_output, prime_sample_batched,
                                                     fixed_guesses_num=_MAX_GUESSES_NUM,
                                                     postfix=f'_THRE{str(self.prime_model.preds_filter.endpt_dist)}')
                prime_fde_list = get_fde_statistic(self.prime_dataset_val, self.args.pred_len, prime_filter_output, prime_sample_batched)


                rule_filter_output, rule_filter_prbs, rule_nofilter_output, rule_sample_batched = \
                    self.rule_prime_model.post_process(self.rule_prime_dataset_val, rule_output, rule_input, phase='validate')
                rule_metrics = get_accuracy_statistic(self.rule_prime_dataset_val, self.args.pred_len, rule_nofilter_output, rule_sample_batched)
                rule_metrics_flt = get_accuracy_statistic(self.rule_prime_dataset_val, self.args.pred_len, rule_filter_output, rule_sample_batched,
                                                     fixed_guesses_num=_MAX_GUESSES_NUM,
                                                     postfix=f'_THRE{str(self.rule_prime_model.preds_filter.endpt_dist)}')
                rule_fde_list = get_fde_statistic(self.rule_prime_dataset_val, self.args.pred_len, rule_filter_output, rule_sample_batched)

                prime_sample_batched = trans_back_sample_batched(prime_sample_batched)
                rule_sample_batched = trans_back_sample_batched(rule_sample_batched)

                axes = plt.subplot(111)
                axes = draw_lanelet_map(hd_map.lanelet_map, axes)

                
                min_x = 1e9
                min_y = 1e9
                max_x = -1e9
                max_y = -1e9
                alpha_num = 0.6
                line_width = 2
                # plot DenseTNT
                dense_tnt_traj = dense_output[0]
                for i in range(dense_tnt_traj.shape[0]):
                    if i != 0:
                        axes.plot(dense_tnt_traj[i, :, 0], dense_tnt_traj[i, :, 1], color="#7E2065", linewidth=line_width, alpha=alpha_num)
                        axes.scatter(dense_tnt_traj[i, -1, 0], dense_tnt_traj[i, -1, 1], color="#7E2065", marker="o", s=20)
                    else:
                        axes.plot(dense_tnt_traj[i, :, 0], dense_tnt_traj[i, :, 1], color="#7E2065", linewidth=line_width, alpha=alpha_num, label="DenseTNT")
                        axes.scatter(dense_tnt_traj[i, -1, 0], dense_tnt_traj[i, -1, 1], color="#7E2065", marker="o", s=20)
                        
                    if np.min(dense_tnt_traj[i, :, 0]) < min_x:
                        min_x = np.min(dense_tnt_traj[i, :, 0])

                    if np.min(dense_tnt_traj[i, :, 1]) < min_y:
                        min_y = np.min(dense_tnt_traj[i, :, 1])

                    if np.max(dense_tnt_traj[i, :, 0]) > max_x:
                        max_x = np.max(dense_tnt_traj[i, :, 0])

                    if np.max(dense_tnt_traj[i, :, 1]) > max_y:
                        max_y = np.max(dense_tnt_traj[i, :, 1])
                # plot PRIME
                prime_pred_traj = prime_filter_output[seq_id]
                for i in range(prime_pred_traj.shape[0]):
                    if i!=0:
                        axes.plot(prime_pred_traj[i, :, 0], prime_pred_traj[i, :, 1], color="#144A74", linewidth=line_width, alpha=alpha_num)
                        axes.scatter(prime_pred_traj[i, -1, 0], prime_pred_traj[i, -1, 1], color="#144A74", marker="o", s=20)
                    else:
                        axes.plot(prime_pred_traj[i, :, 0], prime_pred_traj[i, :, 1], color="#144A74", linewidth=line_width, alpha=alpha_num, label="PRIME")
                        axes.scatter(prime_pred_traj[i, -1, 0], prime_pred_traj[i, -1, 1], color="#144A74", marker="o", s=20)

                    if np.min(prime_pred_traj[i, :, 0]) < min_x:
                        min_x = np.min(prime_pred_traj[i, :, 0])

                    if np.min(prime_pred_traj[i, :, 1]) < min_y:
                        min_y = np.min(prime_pred_traj[i, :, 1])

                    if np.max(prime_pred_traj[i, :, 0]) > max_x:
                        max_x = np.max(prime_pred_traj[i, :, 0])

                    if np.max(prime_pred_traj[i, :, 1]) > max_y:
                        max_y = np.max(prime_pred_traj[i, :, 1])

                # plot Rule PRIME
                rule_pred_traj = rule_filter_output[seq_id]
                for i in range(rule_pred_traj.shape[0]):
                    if i != 0:
                        axes.plot(rule_pred_traj[i, :, 0], rule_pred_traj[i, :, 1], color="#C21F30", linewidth=line_width, alpha=alpha_num)
                        axes.scatter(rule_pred_traj[i, -1, 0], rule_pred_traj[i, -1, 1], color="#C21F30", marker="o", s=20)
                    else:
                        axes.plot(rule_pred_traj[i, :, 0], rule_pred_traj[i, :, 1], color="#C21F30", linewidth=line_width, alpha=alpha_num, label="Rule-PRIME")
                        axes.scatter(rule_pred_traj[i, -1, 0], rule_pred_traj[i, -1, 1], color="#C21F30", marker="o", s=20)

                    if np.min(rule_pred_traj[i, :, 0]) < min_x:
                        min_x = np.min(rule_pred_traj[i, :, 0])

                    if np.min(rule_pred_traj[i, :, 1]) < min_y:
                        min_y = np.min(rule_pred_traj[i, :, 1])

                    if np.max(rule_pred_traj[i, :, 0]) > max_x:
                        max_x = np.max(rule_pred_traj[i, :, 0])

                    if np.max(rule_pred_traj[i, :, 1]) > max_y:
                        max_y = np.max(rule_pred_traj[i, :, 1])
                
                # plot obs gt
                obs_xy = prime_sample_batched["agent_obs_xy"][0]
                gt_xy = prime_sample_batched["agent_gt_xy"][0]

                # prime_pred_traj = prime_filter_output[seq_id]
                prime_minFDE_6, prime_minADE_6, _ = calcu_min_ade_fde(prime_pred_traj, gt_xy)
                rule_minFDE_6, rule_minADE_6, _ = calcu_min_ade_fde(rule_pred_traj, gt_xy)

                axes.plot(obs_xy[:, 0], obs_xy[:, 1], color="#FC8C23", linewidth=line_width, label="Obs", zorder=10)
                axes.scatter(obs_xy[-1, 0], obs_xy[-1, 1], color="#FC8C23", marker="x", s=20, zorder=10)

                axes.plot(gt_xy[:, 0], gt_xy[:, 1], color="#207F4C", linewidth=line_width, label="GTs", zorder=10)
                axes.scatter(gt_xy[-1, 0], gt_xy[-1, 1], color="#207F4C", marker="x", s=20, zorder=10)

                if np.min(obs_xy[:, 0]) < min_x:
                        min_x = np.min(obs_xy[:, 0])

                if np.min(obs_xy[:, 1]) < min_y:
                    min_y = np.min(obs_xy[:, 1])

                if np.max(obs_xy[:, 0]) > max_x:
                    max_x = np.max(obs_xy[:, 0])

                if np.max(obs_xy[:, 1]) > max_y:
                    max_y = np.max(obs_xy[:, 1])

                if np.min(gt_xy[:, 0]) < min_x:
                        min_x = np.min(obs_xy[:, 0])

                if np.min(gt_xy[:, 1]) < min_y:
                    min_y = np.min(obs_xy[:, 1])

                if np.max(gt_xy[:, 0]) > max_x:
                    max_x = np.max(obs_xy[:, 0])

                if np.max(gt_xy[:, 1]) > max_y:
                    max_y = np.max(obs_xy[:, 1])

                prime_ade = copy.deepcopy(round(prime_minFDE_6, 3))
                prime_fde = copy.deepcopy(round(prime_minADE_6, 3))
                rule_ade = copy.deepcopy(round(rule_minFDE_6, 3))
                rule_fde = copy.deepcopy(round(rule_minADE_6, 3))
                
                axes.set_aspect(1)
                
                title = "PRIME:ADE="+str(prime_ade)+",FDE="+str(prime_fde)+";Rule-PRIME:ADE="+str(rule_ade)+", FDE="+str(rule_fde)
                # title = f"PRIME:ADE={prime_ade},FDE={prime_fde};Rule-PRIME:ADE={rule_ade}, FDE={rule_fde}"
                # print(prime_)
                plt.title(title)
                plt.xlim(min_x-5, max_x+5)
                plt.ylim(min_y-5, max_y+5)
                plt.xticks([])
                plt.yticks([])
                # plt.legend()
                # print("Hello")
                file_path = os.path.join(self.save_dir, f"{seq_id[:-4]}.svg")
                # print("file name: ",file_path)
                plt.savefig(file_path)
                # plt.show()
                plt.cla()
                plt.clf()
                
                # exit(0)


if __name__ == '__main__':
    prime_data_path = "/home/joe/Desktop/trained_model/prime/feature/val/split"
    rule_prime_data_path = "/home/joe/Desktop/trained_model/rule_prime/best_traj_20/feature/frenet_val/split"

    prime_model_path = "/home/joe/Desktop/trained_model/prime/train/model_best.pth"
    rule_model_path = "/home/joe/Desktop/trained_model/rule_prime/best_traj_20/target_20_normal/train/model_checkpoint_55.pth"

    target_veh_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/val_target_filter"
    map_path = "/home/joe/Dataset/Interaction/INTERACTION-Dataset-DR-single-v1_2/maps"

    args = parse_arguments()
    
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser()
    utils.add_argument(parser)
    dense_args: utils.Args = parser.parse_args()
    utils.init(dense_args, logger)

    compare_predictor = ComparePredictor(
        args, dense_args,
        prime_data_path,
        rule_prime_data_path,
        prime_model_path,
        rule_model_path,
        target_veh_path,
        map_path
    )

    compare_predictor.validate()

    