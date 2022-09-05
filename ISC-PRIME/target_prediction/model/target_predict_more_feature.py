from matplotlib.pyplot import axes
import torch.nn as nn
import torch
import math

from target_prediction.model.backbone.vectornet import VectorNetBackbone
from target_prediction.model.layers.target_pred import TargetPred
from target_prediction.utils.loss import TargetPredLoss
from prime_evaluator.models.basic import TrajEncoder


class TargetPredict(nn.Module):
    def __init__(self,
                 in_channels: int = 8,
                 horizon: int = 30,
                 num_subgraph_layer: int = 3,
                 num_global_graph_layer: int = 1,
                 subgraph_width: int = 64,
                 global_graph_width: int = 64,
                 with_aux: bool = False,
                 aux_width: int = 64,
                 target_pred_hid: int = 64,
                 m: int = 50,
                 k: int = 6,
                 device=torch.device("cuda")):
        """
        Target prediction in TNT algorithm.
        :param in_channels: int, the number of channels of the input node features.
        :param horizon: int, the prediction horizon (prediction length)
        :param num_subgraph_layer: int, the number of subgraph layer
        :param num_global_graph_layer: int, the number of global interaction layer
        :param subgraph_width: int, the channels of extracted subgraph features
        :param global_graph_width: int, the channels of extracted global graph features
        :param with_aux: bool, with aux loss or not
        :param aux_width: int, the hidden dimension of aux recovery mlp
        :param target_pred_hid: int, the hidden dimension of target prediction
        :param m: int, the number of selected candidate
        :param k: int. final output trajectories
        :param device: the device for computation
        """
        super(TargetPredict, self).__init__()

        self.horizon = horizon
        self.m = m
        self.k = k
        self.with_aux = with_aux
        self.device = device

        self.criterion = TargetPredLoss(aux_loss=self.with_aux,
                                        device=self.device)

        # feature extraction backbone
        self.backbone = VectorNetBackbone(
            in_channels=in_channels,
            num_subgraph_layers=num_subgraph_layer,
            num_global_graph_layer=num_global_graph_layer,
            subgraph_width=subgraph_width,
            global_graph_width=global_graph_width,
            aux_mlp_width=aux_width,
            with_aux=with_aux,
            device=device
        )

        traj_feature_num = 64
        self.dim_in = 3
        self.traj_encoder = TrajEncoder(embed_size=32, enc_size=traj_feature_num, norm="GN", dim_in=self.dim_in, act="leaky_relu")

        self.target_pred_layer = TargetPred(
            in_channels=global_graph_width+traj_feature_num,
            hidden_dim=target_pred_hid,
            m=m,
            device=device
        )
        self._init_weight()

    def _init_weight(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, nn.LayerNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def forward(self, data):
        """
        Output prediction for training
        :param data: observed sequence data
        :return:
        """
        n = int(data.candidate_len_max[0].cpu().numpy())

        target_candidate = data.candidate.view(-1, n, 2)    # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()
        candidate_mask = data.candidate_mask.view(-1, n)

        agt_obs_xy = data.agt_obs_xy.view(-1, 10, 2)
        vx_array = data.vx_array.view(-1, 40, 1)
        vy_array = data.vy_array.view(-1, 40, 1)
        yaw_array = data.yaw_array.view(-1, 40, 1)

        # position embedding
        # pe = torch.zeros((10, self.dim_in))
        # position = torch.arange(1, 11).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.dim_in, 2) * -(math.log(10000) / 5))
        # pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        # pe[:, 1::2] = torch.cos(position * div_term[:self.dim_in//2])  # 计算奇数维度的pe值
        # pe = pe.unsqueeze(0)
        # pe = pe.repeat(batch_size, 1, 1)

        if self.dim_in == 5:
            total_input = torch.cat((agt_obs_xy, vx_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, vy_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, yaw_array[:, :10, :]), axis=2)
        else:
            total_input = torch.cat((vx_array[:, :10, :], vy_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, yaw_array[:, :10, :]), axis=2)

        # traj_feature = self.traj_encoder(total_input+pe.cuda())
        traj_feature = self.traj_encoder(total_input)
        traj_feature = traj_feature.unsqueeze(1)

        # feature encoding
        global_feat, aux_out, aux_gt = self.backbone(data)
        target_feat = global_feat[:, 0].unsqueeze(1)

        total_feature = torch.cat((target_feat, traj_feature), dim=2)

        # predict prob. for each target candidate, and corresponding offset
        target_prob, offset = self.target_pred_layer(total_feature, target_candidate, candidate_mask)

        return {
            "target_prob": target_prob,
            "offset": offset
        }, aux_out, aux_gt

    def loss(self, data):
        """
        compute loss according to the gt
        :param data: node feature data
        :return: loss
        """
        n = data.candidate_len_max[0]
        data.y = data.y.view(-1, self.horizon, 2).cumsum(axis=1)

        pred, aux_out, aux_gt = self.forward(data)

        gt = {
            "target_prob": data.candidate_gt.view(-1, n),
            "offset": data.offset_gt.view(-1, 2),
            "y": data.y.view(-1, self.horizon * 2)
        }

        return self.criterion(pred, gt, aux_out, aux_gt)

    def inference(self, data):
        """
        predict the top k most-likely trajectories
        :param data: observed sequence data
        :return:
        """
        n = data.candidate_len_max[0]
        target_candidate = data.candidate.view(-1, n, 2)  # [batch_size, N, 2]
        batch_size, _, _ = target_candidate.size()

        agt_obs_xy = data.agt_obs_xy.view(-1, 10, 2)

        vx_array = data.vx_array.view(-1, 40, 1)
        vy_array = data.vy_array.view(-1, 40, 1)
        yaw_array = data.yaw_array.view(-1, 40, 1)

        # position embedding
        # pe = torch.zeros((10, self.dim_in))
        # position = torch.arange(1, 11).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, self.dim_in, 2) * -(math.log(10000) / 5))
        # pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
        # pe[:, 1::2] = torch.cos(position * div_term[:self.dim_in // 2])  # 计算奇数维度的pe值
        # pe = pe.unsqueeze(0)
        # pe = pe.repeat(batch_size, 1, 1)

        if self.dim_in == 5:
            total_input = torch.cat((agt_obs_xy, vx_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, vy_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, yaw_array[:, :10, :]), axis=2)
        else:
            total_input = torch.cat((vx_array[:, :10, :], vy_array[:, :10, :]), axis=2)
            total_input = torch.cat((total_input, yaw_array[:, :10, :]), axis=2)

        # traj_feature = self.traj_encoder(total_input+pe.cuda())
        traj_feature = self.traj_encoder(total_input)
        traj_feature = traj_feature.unsqueeze(1)

        global_feat, _, _ = self.backbone(data)  # [batch_size, time_step_len, global_graph_width]
        target_feat = global_feat[:, 0].unsqueeze(1)

        total_feature = torch.cat((target_feat, traj_feature), dim=2)

        # predict the prob. of target candidates and selected the most likely M candidate
        target_prob, offset_pred = self.target_pred_layer(total_feature, target_candidate)
        _, indices = target_prob.topk(self.m, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.m)]).T
        target_pred_se, offset_pred_se = target_candidate[batch_idx, indices], offset_pred[batch_idx, indices]

        return target_pred_se, offset_pred_se
