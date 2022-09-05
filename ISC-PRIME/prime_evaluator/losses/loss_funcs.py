import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from prime_evaluator.utils.calcu_utils import DispErrorCalculator, ScoreEvaluator
from prime_evaluator.utils.config import (
    _MAX_GUESSES_NUM,
    _MISSING_THRESHOLD,
    _RANK_LOSS_WEIGHT,
    _LANE_LOSS_WEIGHT,
)


def cross_entropy_between_scores(output_scores, target_scores, reduction: str):
    """
    calculating cross_entropy between tow distributions (intialized by scores)
    """
    target_softmax = F.softmax(target_scores, dim=0)
    output_log_softmax = F.log_softmax(output_scores, dim=0)
    # modulating_factor = (1 - F.softmax(output_scores, dim=0)) ** 2
    # modulating_factor = (target_softmax - F.softmax(output_scores, dim=0)).abs()
    # modulating_factor = (target_softmax - F.softmax(output_scores, dim=0)) ** 2
    modulating_factor = 1.0
    cross_entropy_loss = -target_softmax * modulating_factor * output_log_softmax
    if reduction == 'sum':
        loss = cross_entropy_loss.sum()
    elif reduction=='mean':
        loss = cross_entropy_loss.mean()
    elif reduction=='none':
        loss = cross_entropy_loss
    else:
        raise RuntimeError("Wrong type of reduction is specified [mean/sum/none]")

    return loss


# class PredRank(nn.Module):
#     """
#     Pure Ranking loss: max(sum( Ind(p>p*) ) - MAX_GUESSES_NUM, 0)
#     """
#     def __init__(self, args):
#         super(PredRank, self).__init__()
#         self.args = args
#
#     def forward(self, output, sample_batched):
#         cls_idx = sample_batched['cls_start_end_idx']
#         trajs_idx = sample_batched['trajs_start_end_idx']
#         traj_oracle_idx = sample_batched['fut_oracle_idx']
#         batch_losses = torch.Tensor().requires_grad_(True).to(output.device)
#         for sample_id, cls_start_end in enumerate(cls_idx):
#             first = trajs_idx[cls_start_end[0]][0]
#             last = trajs_idx[cls_start_end[1]-1][1]
#             sample_output = output[first:last]
#             oracle_id = traj_oracle_idx[sample_id]
#             indicator = (sample_output > sample_output[oracle_id]).float()
#             topK_ranking_loss = max((torch.sum(indicator)-_MAX_GUESSES_NUM) / indicator.shape[0],
#                                     torch.tensor(0.0, device=indicator.device))
#             batch_losses = torch.cat((batch_losses, topK_ranking_loss.unsqueeze(0)), 0)
#         return batch_losses.mean()


class PredClassify(nn.Module):
    """
    Cross Entropy between the predicted trajectories and the oracle one
    """
    def __init__(self, args, with_rank_loss = True):
        super(PredClassify, self).__init__()
        self.args = args
        self.with_rank_loss = with_rank_loss
        self.rank_weight = _RANK_LOSS_WEIGHT
        logging.info(f"Classification Loss -- **RankLoss: {self.with_rank_loss} (weight={self.rank_weight})")

    def forward(self, output, sample_batched):
        cls_idx = sample_batched['cls_start_end_idx']
        trajs_idx = sample_batched['trajs_start_end_idx']
        traj_oracle_idx = sample_batched['fut_oracle_idx']

        batch_losses = torch.Tensor().requires_grad_(True).to(output.device)
        for sample_id, cls_start_end in enumerate(cls_idx):
            first = trajs_idx[cls_start_end[0]][0]
            last = trajs_idx[cls_start_end[1]-1][1]
            sample_output = output[first:last]
            oracle_id = traj_oracle_idx[[sample_id]]
            # "cross_entropy" combines log_softmax and nll_loss in a single function.
            classification_loss = F.cross_entropy(input=sample_output.view(1,-1), target=oracle_id.long())
            if self.with_rank_loss:
                indicator = (sample_output > sample_output[oracle_id[0]]).float()
                topK_ranking_loss = max((torch.sum(indicator) - _MAX_GUESSES_NUM) / sample_output.shape[0] * self.rank_weight,
                                        torch.tensor(0.0, device=indicator.device))
                classification_loss = classification_loss + topK_ranking_loss
            batch_losses = torch.cat((batch_losses, classification_loss.unsqueeze(0)), 0)
        return batch_losses.mean()


class DualClassify(nn.Module):
    """
    Cross Entropy between in lane-level and trajectory-level
    """
    def __init__(self, args, type):
        super(DualClassify, self).__init__()
        self.args = args
        self.temp = args.score_temp
        self.lane_weight = _LANE_LOSS_WEIGHT
        self.type = type
        if type == 'score':
            self.score_evaluator = ScoreEvaluator(args.dist_metric, args.dist_weighted)
            logging.info(f"Scoring Loss -- temp={self.temp} weighted={args.dist_weighted}")
        elif type == 'regress':
            self.disp_calculator = DispErrorCalculator(args.dist_metric)
            logging.info(f"Regress Loss -- dist calculated {args.dist_metric}")

    def forward(self, output, sample_batched):
        if not (isinstance(output, list) and len(output)==2):
            assert False, "Mismatched type of output for DualClassify loss"
        cls_idx = sample_batched['cls_start_end_idx']
        trajs_idx = sample_batched['trajs_start_end_idx']
        pred_candidates = sample_batched['agent_futs_xy']
        pred_gt = sample_batched['agent_gt_xy']
        scales = sample_batched['scales']
        cls_oracle_bool = sample_batched['agent_cls_oracle']

        lane_scores, traj_scores = output
        batch_losses = torch.Tensor().requires_grad_(True).to(output[0].device)
        for sample_id, cls_start_end in enumerate(cls_idx):
            # Lane loss: calculating between all the probability and the one(s) marked as oracle
            cls_id_first, cls_id_last = cls_start_end
            lane_target = cls_oracle_bool[cls_id_first:cls_id_last].float()
            lane_target = lane_target / lane_target.sum()
            lane_output = lane_scores[cls_id_first:cls_id_last].view(1, -1)
            lane_loss = torch.sum(-lane_target * F.log_softmax(lane_output, dim=-1)) * self.lane_weight
            # Traj loss: calculating the predicted trajs lies inside the oracle lanes
            traj_loss = []
            for cls_id in range(cls_id_first, cls_id_last):
                if cls_oracle_bool[cls_id]:
                    traj_id_first, traj_id_last = trajs_idx[cls_id]
                    scores_output = traj_scores[traj_id_first:traj_id_last].squeeze(1)
                    futs_under_this_lane = pred_candidates[traj_id_first:traj_id_last]
                    scores = self.score_evaluator.scoring_trajs(futs_under_this_lane, pred_gt[sample_id], scale=scales[sample_id])
                    traj_loss.append(cross_entropy_between_scores(scores_output, scores / self.temp))
                    ## To check distribution:
                    # np.vstack((F.softmax(scores).cpu().numpy(), F.softmax(scores/0.1).cpu().numpy(), F.softmax(scores/0.01).cpu().numpy())).transpose()
            batch_losses = torch.cat((batch_losses, (lane_loss + sum(traj_loss)/len(traj_loss)).unsqueeze(0)), 0)
        return batch_losses.mean()


class PredRegress(nn.Module):
    """
    Regress from the corresponding predicted trajectory
    """
    def __init__(self, args):
        super(PredRegress, self).__init__()
        self.args = args
        # The loss calculation is specified by dist metric.
        self.disp_calculator = DispErrorCalculator(args.dist_metric)
        logging.info(f"Regress Loss -- dist calculated {args.dist_metric}")

    def forward(self, output, sample_batched, apply_lane_mask = False):
        batch_size = len(sample_batched['seq_ids'])
        cl_start_end_idx = sample_batched['cls_start_end_idx']
        trajs_start_end_idx = sample_batched['trajs_start_end_idx']
        cl_oracles = sample_batched['agent_cls_oracle']

        preds_count_under_lane = [start_end[1] - start_end[0] for start_end in trajs_start_end_idx]
        preds_count_under_sample = [sum(preds_count_under_lane[start_end[0]:start_end[1]]) for start_end in cl_start_end_idx]

        batch_losses = torch.Tensor().requires_grad_(True).to(output.device)
        for i in range(batch_size):
            cls_range = cl_start_end_idx[i]
            first = trajs_start_end_idx[cls_range[0]][0]
            last = trajs_start_end_idx[cls_range[1] - 1][1]
            sample_preds = sample_batched['agent_futs_xy'][first:last]
            sample_gt = sample_batched['agent_gt_xy'][i]
            dist_error_gt = self.disp_calculator.calc_disp_error(sample_preds, sample_gt)
            dist_error_output = output[first:last].squeeze(1)
            # delta_error = torch.pow(dist_error_output - dist_error_gt, 2)   # L2 loss
            delta_error = torch.abs(dist_error_output - dist_error_gt)   # L1 loss
            if apply_lane_mask:
                mask = torch.cat([ torch.tensor([True]*preds_count_under_lane[cls])
                                   if cl_oracles[cls] else
                                   torch.tensor([False]*preds_count_under_lane[cls])
                                   for cls in range(cls_range[0], cls_range[1])
                                   ]).float().to(output.device)
                regress_loss = (delta_error * mask).sum() / mask.sum()
            else:
                regress_loss = delta_error.mean()
            batch_losses = torch.cat((batch_losses, regress_loss.unsqueeze(0)), 0)
        return batch_losses.mean()


class ScoreClassify(nn.Module):
    """
    Cross Entropy between the predicted scores and ground truth scores.
    """
    def __init__(self, args, with_rank_loss = False, with_miss_loss = True):
        super(ScoreClassify, self).__init__()
        self.args = args
        self.with_rank_loss = with_rank_loss
        self.with_miss_loss = with_miss_loss
        self.temp = args.score_temp

        # Loss weights
        self.classify_weight = 1.0  # Set to 1 when cross_entropy calculation is reduced by "sum"
        self.rank_weight = _RANK_LOSS_WEIGHT
        self.miss_weight = 1.0

        self.score_evaluator = ScoreEvaluator(args.dist_metric, args.dist_weighted)
        logging.info(f"Scoring Loss -- temp={self.temp} weightedDist?{args.dist_weighted}\n"
                     f"ClassificationLoss: weight={self.classify_weight}\n"
                     f"**RankLoss({self.with_rank_loss}): weight={self.rank_weight}\n"
                     f"**MissLoss({self.with_miss_loss}): weight={self.miss_weight}")

    def forward(self, output, sample_batched):
        cls_idx = sample_batched['cls_start_end_idx']
        trajs_idx = sample_batched['trajs_start_end_idx']
        pred_candidates = sample_batched['agent_futs_xy']
        pred_gt = sample_batched['agent_gt_xy']
        scales = sample_batched['scales']
        traj_oracle_idx = sample_batched['fut_oracle_idx']

        batch_losses = torch.Tensor().requires_grad_(True).to(output.device)
        topK_ranking_loss = topK_missing_loss = 0.0
        for sample_id, cls_start_end in enumerate(cls_idx):
            first = trajs_idx[cls_start_end[0]][0]
            last = trajs_idx[cls_start_end[1]-1][1]
            sample_output = output[first:last].squeeze(1)
            sample_futs = pred_candidates[first:last]
            scores = self.score_evaluator.scoring_trajs(sample_futs, pred_gt[sample_id], scale=scales[sample_id])
            classification_loss = cross_entropy_between_scores(sample_output, scores / self.temp, reduction='sum')
            sample_loss = self.classify_weight * classification_loss

            if self.with_rank_loss:
                oracle_id = traj_oracle_idx[sample_id]
                indicator = (sample_output > sample_output[oracle_id]).float()
                topK_ranking_loss = max((torch.sum(indicator) - _MAX_GUESSES_NUM) / sample_output.shape[0], torch.tensor(0.0, device=output.device))
                sample_loss += self.rank_weight * topK_ranking_loss

            if self.with_miss_loss:
                topK_futs = sample_futs[torch.topk(sample_output, min(sample_output.shape[0], _MAX_GUESSES_NUM)).indices]
                topK_disp = torch.norm(scales[sample_id] * (topK_futs - pred_gt[sample_id]), dim=-1)
                topK_missing_loss = max(min(topK_disp[:, -1]) - _MISSING_THRESHOLD, torch.tensor(0.0, device=output.device))
                sample_loss += self.miss_weight * topK_missing_loss

            batch_losses = torch.cat((batch_losses, sample_loss.unsqueeze(0)), 0)
            # print("clf={:.2f}\t rank={:.2f}\t miss={:.2f}".format(
            #     self.classify_weight * classification_loss, self.rank_weight * topK_ranking_loss, self.miss_weight * topK_missing_loss))

        return batch_losses.mean()