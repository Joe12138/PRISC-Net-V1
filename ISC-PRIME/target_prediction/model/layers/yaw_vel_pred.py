import torch
import torch.nn as nn
import torch.nn.functional as F

from target_prediction.model.layers.basic_module import MLP
from target_prediction.model.layers.utils import masked_softmax


class YawVelPred(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dim: int = 64,
                 device=torch.device("cuda")):
        super(YawVelPred, self).__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.M = 6

        # self.variable = variable

        self.device = device

        self.yaw_prob_mlp = nn.Sequential(
            MLP(
                in_channel=in_channels+1,
                out_channel=hidden_dim,
                hidden=hidden_dim
            ),
            nn.Linear(hidden_dim, 1)
        )

        self.mean_mlp = nn.Sequential(
            MLP(
                in_channel=in_channels+1,
                out_channel=hidden_dim,
                hidden=hidden_dim
            ),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, feat_in: torch.Tensor, yaw_candidate: torch.Tensor):
        assert feat_in.dim() == 3, "[YawPred]: Error input feature dimension"

        batch_size, n, _ = yaw_candidate.size()

        # stack the target candidate to the end of input feature
        feat_in_repeat = torch.cat([feat_in.repeat(1, n, 1), yaw_candidate], dim=2)

        prob_tensor = self.yaw_prob_mlp(feat_in_repeat).squeeze(2)

        yaw_candit_prob = F.softmax(prob_tensor, dim=-1)
        yaw_offset_mean = self.mean_mlp(feat_in_repeat)

        return yaw_candit_prob, yaw_offset_mean

    def loss(self,
             feat_in: torch.Tensor,
             yaw_candidate: torch.Tensor,
             yaw_gt: torch.Tensor,
             yaw_offset_gt: torch.Tensor):
        batch_size, n, _ = yaw_candidate.size()
        _, num_cand = yaw_gt.size()

        assert num_cand == n, "The num target candidate and the ground truth one-hot vector is not aligned: {} vs {};".format(
            n, num_cand)

        yaw_candit_prob, yaw_offset_mean = self.forward(feat_in, yaw_candidate)
        n_candidate_loss = F.cross_entropy(yaw_candit_prob.transpose(1, 2), yaw_gt.long(), reduction='sum')

        _, indices = yaw_candit_prob[:, :, 1].topk(self.M, dim=1)
        batch_idx = torch.vstack([torch.arange(0, batch_size, device=self.device) for _ in range(self.M)]).T
        offset_loss = F.smooth_l1_loss(yaw_offset_mean[yaw_gt.bool()], yaw_offset_gt, reduction='sum')

        return n_candidate_loss + offset_loss, yaw_candidate[batch_idx, indices], yaw_offset_mean[batch_idx, indices]

    def inference(self,
                  feat_in: torch.Tensor,
                  tar_candidate: torch.Tensor,):
        """
        output only the M predicted probability of the predicted target
        :param feat_in:        the encoded trajectory features, [batch_size, inchannels]
        :param tar_candidate:  tar_candidate: the target position candidate (x, y), [batch_size, N, 2]
        :param candidate_mask: the mask of valid target candidate
        :return:
        """

        return self.forward(feat_in, tar_candidate)

