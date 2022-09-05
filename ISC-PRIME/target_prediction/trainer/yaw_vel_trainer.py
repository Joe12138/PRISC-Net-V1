from target_prediction.trainer.trainer import Trainer
from target_prediction.model.yaw_vel_predict_more_feature import YawVelPredict
from target_prediction.utils.loss import TargetPredLoss
from target_prediction.utils.optim_schedule import ScheduledOptim

import os
import torch
import numpy as np
from apex import amp
from tqdm import tqdm
from apex.parallel import DistributedDataParallel
from torch.optim import AdamW

from util_dir.geometry import normalize_angle


class YawVelTrainer(Trainer):
    """
    TargetPred Trainer, train the TargetPred with specified hyper parameters and configuration
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 number_workers: int = 1,
                 num_global_graph_layer: int = 1,
                 horizon: int = 30,
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=30,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 multi_gpu=False,
                 enable_log=True,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True,
                 variable: str = "yaw"):
        super(YawVelTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=number_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            multi_gpu=multi_gpu,
            enable_log=enable_log,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose,
            variable=variable
        )

        # init or load model
        self.horizon = horizon
        self.aux_loss = aux_loss

        # input dim: (20, 8); output dim: (30, 2)
        model_name = YawVelPredict
        self.model = model_name(
            in_channels=self.trainset.num_features if hasattr(self.trainset, "num_features") else self.testset.num_features,
            horizon=self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device,
            variable=variable
        )

        self.criterion = TargetPredLoss(
            aux_loss=self.model.with_aux,
            device=self.device
        )

        # init optimizer
        self.optim = AdamW(params=self.model.parameters(),
                           lr=self.lr,
                           betas=self.betas,
                           weight_decay=self.weight_decay)
        self.optim_schedule = ScheduledOptim(
            optimizer=self.optim,
            init_lr=self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # resume from file or maintain the original
        if model_path:
            self.load(model_path, "m")

        self.model = self.model.to(self.device)
        if self.multi_gpu:
            self.model = DistributedDataParallel(self.model)
            self.model, self.optimizer = amp.initialize(self.model, self.optim, opt_level="O0")
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0)):
                print("[TargetPredTrainer]: Train the model with multiple GPUs: {} GPUs.".format(int(os.environ["WORLD_SIZE"])))
        else:
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0)):
                print("[TargetPredTrainer]: Train the model with single device on {}.".format(self.device))

        # record the init learning rate
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0):
            self.write_log("LR", self.lr, 0)

        # resume training from ckpt
        if ckpt_path:
            self.load(ckpt_path, "c")

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                0.0,
                avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            n_graph = data.num_graphs
            data = data.to(self.device)

            if training:
                self.optim_schedule.zero_grad()

                if self.multi_gpu:
                    n = 21
                    data.y = data.y.view(-1, self.horizon, 2).cumsum(axis=1)
                    pred, aux_out, aux_gt = self.model(data)

                    gt = {
                        "target_prob": data.yaw_candts_gt.view(-1, n),
                        "offset": data.yaw_offset_gt.view(-1, 1),
                        "y": data.y.view(-1, self.horizon * 2)
                    }

                    loss, loss_dict = self.criterion(pred, gt, aux_out, aux_gt)
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()

                else:
                    loss, loss_dict = self.model.loss(data)
                    loss.backward()

                self.optim.step()

                # writing loss
                # if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0):
                #     self.write_log("Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                #     self.write_log("Target_Cls_Loss",
                #                 loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                #     self.write_log("Target_Offset_Loss",
                #                 loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                #     self.write_log("Traj_Loss",
                #                 loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                #     self.write_log("Score_Loss",
                #                 loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    if self.multi_gpu:
                        n = 21
                        data.y = data.y.view(-1, self.horizon, 2).cumsum(axis=1)
                        pred, aux_out, aux_gt = self.model(data)

                        gt = {
                            "target_prob": data.yaw_candts_gt.view(-1, n),
                            "offset": data.yaw_offset_gt.view(-1, 1),
                            "y": data.y.view(-1, self.horizon * 2)
                        }

                        loss, loss_dict = self.criterion(pred, gt, aux_out, aux_gt)
                    else:
                        loss, loss_dict = self.model.loss(data)

                    # writing loss
                    if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0):
                        self.write_log("Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                loss.detach().item() / n_graph,
                avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 0):
                learning_rate = self.optim_schedule.step_and_update_lr()
                self.write_log("LR", learning_rate, epoch + 1)

        return avg_loss / num_sample

    def test(self):
        self.model.eval()

        convert_coordinate = False

        min_fde = 10e9
        max_fde = -10e9
        avg_fde = 0
        index = 0
        fde_list = []

        # all_ade = self.compute_yaw_metric()

        # print(all_ade)

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs

                if self.variable == "yaw":
                    gt = data.yaw_array.unsqueeze(1).view(batch_size, -1, 1).numpy()
                elif self.variable == "vx":
                    gt = data.vx_array.unsqueeze(1).view(batch_size, -1, 1).numpy()
                else:
                    gt = data.vy_array.unsqueeze(1).view(batch_size, -1, 1).numpy()

                out = self.model.inference(data.to(self.device))

                target_pred_se, offset_pred_se = out
                target_pred_se = target_pred_se.cpu().numpy()
                offset_pred_se = offset_pred_se.cpu().numpy()

                for batch_id in range(batch_size):
                    # cur_batch_traj = target_pred_se[batch_id]

                    if self.variable == "yaw":
                        real_coord_pred = []
                        for i in range(target_pred_se[batch_id].shape[0]):
                            real_coord_pred.append(normalize_angle(target_pred_se[batch_id][i]+offset_pred_se[batch_id][i]))
                            # real_coord_pred = [normalize_angle(pred_y_k) for pred_y_k in target_pred_se[batch_id]+offset_pred_se[batch_id]]
                    else:
                        real_coord_pred = [pred_y_k for pred_y_k in target_pred_se[batch_id]+offset_pred_se[batch_id]]
                    gt_trajectories = gt[batch_id]

                    batch_fde = 10e9
                    for i in range(1):
                        diff = abs(real_coord_pred[i]-gt_trajectories[-1])

                        if diff < batch_fde:
                            batch_fde = diff

                    if batch_fde < min_fde:
                        min_fde = batch_fde
                    if batch_fde > max_fde:
                        max_fde = batch_fde

                    avg_fde += batch_fde
                    fde_list.append(batch_fde)
                    index += 1

        print("min_fde = {}, max_fde = {}, avg_fde = {}".format(min_fde, max_fde, np.mean(fde_list)))
        print("var = {}".format(np.var(fde_list)))

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted
