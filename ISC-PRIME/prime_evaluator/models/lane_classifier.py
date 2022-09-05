import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.special import softmax
from prime_evaluator.utils.calcu_utils import PreditionsFilter
from prime_evaluator.utils.config import (
    _MAX_GUESSES_NUM
)

from prime_evaluator.models.basic import (
    LaneEncoder,
    TrajEncoder,
    ScoreDecoder,
    CoreAttention,
)

class TrajDecoder(nn.Module):
    def __init__(self, input_feature_size):
        super(TrajDecoder, self).__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(input_feature_size, input_feature_size),
            nn.LeakyReLU(0.1),
            nn.Linear(input_feature_size, input_feature_size),
            nn.LeakyReLU(0.1),
            nn.Linear(input_feature_size, 1)
        )

    def forward(self, x):
        scoring = self.fully_connected(x)
        return scoring

###################################################################################
########################### Architecture ##########################################
###################################################################################

class LaneTrajClassifier(nn.Module):

    def __init__(self, args):
        super(LaneTrajClassifier, self).__init__()
        self.use_cuda = args.use_cuda
        self.loss_name = args.loss_name

        logging.info("<Attention(A2L, A2A, F2L, F2F)-prepost>")
        self.preds_filter = PreditionsFilter(2.7, 0.0)

        # Parameters
        self.activation = 'leaky_relu'
        self.lane_embed_size = 16
        self.lane_enc_size = 64
        self.traj_embed_size = 32
        self.traj_enc_size = 64
        self.preds_embed_size = 32
        self.preds_enc_size = 128  # pred get larger structure
        self.lane_combin_size = self.traj_enc_size * 2
        self.traj_combin_size = self.traj_enc_size * 2 + self.preds_enc_size

        ## Feature encoding
        self.cls_encoder = LaneEncoder(self.lane_embed_size, self.lane_enc_size, activation=self.activation,  num_layers=1, bidirectional=True)
        self.nbrs_encoder = TrajEncoder(self.traj_embed_size, self.traj_enc_size, activation=self.activation, dim_in=3, num_layers=1)  # nbrs has padding dim
        self.obs_encoder = TrajEncoder(self.traj_embed_size, self.traj_enc_size, activation=self.activation, dim_in=2, num_layers=1)
        self.pred_encoder = TrajEncoder(self.preds_embed_size, self.preds_enc_size, activation=self.activation, dim_in=2, num_layers=1)

        self.lane_score_decoder = TrajDecoder(self.lane_combin_size)
        self.traj_score_decoder = TrajDecoder(self.traj_combin_size)

        # Mapping feature to query / keys
        self.cls_feature_mapping = nn.Linear(self.lane_enc_size, self.lane_enc_size)
        self.obs_feature_mapping = nn.Linear(self.traj_enc_size, self.traj_enc_size)
        self.nbrs_feature_mapping = nn.Linear(self.traj_enc_size, self.traj_enc_size)
        self.pred_feature_mapping = nn.Linear(self.preds_enc_size, self.traj_enc_size)

        self.social_feature_mapping = nn.Linear(self.traj_enc_size, self.traj_enc_size)
        self.allpreds_feature_mapping = nn.Linear(self.preds_enc_size, self.traj_enc_size)  # Shrink pred's map to the same size with traj

        # Core attention funcs
        self.l2a_attention = CoreAttention(self.traj_enc_size, self.lane_enc_size, self.lane_enc_size)
        self.a2a_attention = CoreAttention(self.traj_enc_size, self.traj_enc_size, self.traj_enc_size)
        self.l2f_attention = CoreAttention(self.preds_enc_size, self.lane_enc_size, self.lane_enc_size)
        self.f2f_attention = CoreAttention(self.preds_enc_size, self.traj_enc_size, self.preds_enc_size)


    def forward(self, sample_batched):
        batch_size = len(sample_batched['seq_ids'])
        agent_cls, agent_obs_xy, agent_futs_xy = \
            sample_batched['agent_cls'], sample_batched['agent_obs_xy'], sample_batched['agent_futs_xy']
        nbrs_obs_xy, nbrs_obs_padding, nbrs_start_end_idx = \
            sample_batched['nbrs_obs_xy'], sample_batched['nbrs_obs_padding'], sample_batched['nbrs_start_end_idx']
        nbrs_obs_xyp = torch.cat((nbrs_obs_xy, nbrs_obs_padding.unsqueeze(2)), -1)
        cl_start_end_idx, trajs_start_end_idx = \
            sample_batched['cls_start_end_idx'], sample_batched['trajs_start_end_idx']

        obs_encoding = self.obs_encoder(agent_obs_xy)  # num_samples x traj_enc_size
        obs_mapping = self.obs_feature_mapping(obs_encoding)
        cls_encoding = self.cls_encoder(agent_cls)  # total_cls x num_parts x lane_embed_size
        cls_mapping = self.cls_feature_mapping(cls_encoding)
        nbrs_encoding = self.nbrs_encoder(nbrs_obs_xyp)  # total_nbrs x traj_enc_size
        nbrs_mapping = self.nbrs_feature_mapping(nbrs_encoding)
        pred_encoding = self.pred_encoder(agent_futs_xy)  # total_preds x traj_enc_size
        pred_mapping = self.pred_feature_mapping(pred_encoding)

        nbrs_count_under_sample = [start_end[1] - start_end[0] for start_end in nbrs_start_end_idx]
        preds_count_under_lane = [start_end[1] - start_end[0] for start_end in trajs_start_end_idx]
        preds_count_under_sample = [sum(preds_count_under_lane[start_end[0]:start_end[1]]) for start_end in cl_start_end_idx]

        # Batch loop
        lane_level_combin = torch.empty((0, self.lane_combin_size)).cuda() if self.use_cuda else torch.empty((0, self.lane_combin_size))
        traj_level_combin = torch.empty((0, self.traj_combin_size)).cuda() if self.use_cuda else torch.empty((0, self.traj_combin_size))
        for i in range(batch_size):
            ## Transform all encs to [batch(pred_num) x num_obj x feature_dim]
            obs_encs = obs_encoding[[i]].repeat(preds_count_under_sample[i], 1).unsqueeze(1)
            obs_maps = obs_mapping[[i]].repeat(preds_count_under_sample[i], 1).unsqueeze(1)
            cls_range = cl_start_end_idx[i]
            cls_encs = torch.cat(
                [cls_encoding[cls_id].unsqueeze(0).repeat(preds_count_under_lane[cls_id], 1, 1) for cls_id in
                 range(cls_range[0], cls_range[1])], dim=0)
            cls_maps = torch.cat(
                [cls_mapping[cls_id].unsqueeze(0).repeat(preds_count_under_lane[cls_id], 1, 1) for cls_id in
                 range(cls_range[0], cls_range[1])], dim=0)
            nbr_range = nbrs_start_end_idx[i]
            nbr_encs = nbrs_encoding[nbr_range[0]:nbr_range[1]].unsqueeze(0).repeat(preds_count_under_sample[i], 1, 1)
            nbr_maps = nbrs_mapping[nbr_range[0]:nbr_range[1]].unsqueeze(0).repeat(preds_count_under_sample[i], 1, 1)
            pred_encs = pred_encoding[
                        trajs_start_end_idx[cls_range[0]][0]:trajs_start_end_idx[cls_range[1] - 1][1]].unsqueeze(1)
            pred_maps = pred_mapping[
                        trajs_start_end_idx[cls_range[0]][0]:trajs_start_end_idx[cls_range[1] - 1][1]].unsqueeze(1)

            ## allObs-lane (A2L) attention:
            allobs_encs = torch.cat((nbr_encs, obs_encs), dim=1)
            allobs_maps = torch.cat((nbr_maps, obs_maps), dim=1)
            allobs_lane_output = self.l2a_attention(allobs_encs,
                                                    allobs_maps, cls_maps, cls_encs)

            ## Social (A2A) attention
            allobs_lane_output_map = self.social_feature_mapping(allobs_lane_output)
            social_output = self.a2a_attention(allobs_lane_output,
                                               allobs_lane_output_map, allobs_lane_output_map, allobs_lane_output)

            ## Futs-lane (F2L) attention:
            pred_lane_output = self.l2f_attention(pred_encs, pred_maps, cls_maps, cls_encs)

            ## Futs-Futs (F2F) attention:
            pred_lane_output = pred_lane_output.transpose(0, 1)
            pred_lane_output_map = self.allpreds_feature_mapping(pred_lane_output)
            allpreds_output = self.f2f_attention(pred_lane_output,
                                                 pred_lane_output_map, pred_lane_output_map, pred_lane_output)

            ## Merge together
            agent_obs_part = allobs_lane_output[:, [-1], :]
            agent_interaction_part = social_output[:, [-1], :]
            agent_pred_part = allpreds_output.transpose(0, 1)

            first_under_each_lane = [sum(preds_count_under_lane[cls_range[0]:cls_id]) for cls_id in range(cls_range[0], cls_range[1])]
            lanes_combin_this_sample = torch.cat((agent_obs_part[first_under_each_lane], agent_interaction_part[first_under_each_lane]), dim=-1)
            trajs_combin_this_sample = torch.cat((agent_obs_part, agent_interaction_part, agent_pred_part), dim=-1)

            # ## Merge together
            lane_level_combin = torch.cat((lane_level_combin, lanes_combin_this_sample.squeeze(1)), dim=0)
            traj_level_combin = torch.cat((traj_level_combin, trajs_combin_this_sample.squeeze(1)), dim=0)

        output = [self.lane_score_decoder(lane_level_combin),
                  self.traj_score_decoder(traj_level_combin)]           # num_lanes x 1, num_predictions x 1
        return output


    def post_process(self, dataset, output, sample_batched,
                     generate_traj: bool = True,
                     phase: str = 'train'):
        """
        Network output may be further processed, and take the full sample_batched back to CPU
        phase: here it corresponds to the function of train() validate() submit() in predictor file.
        """
        assert phase in ['train', 'validate', 'submit'], "Wrong type of argu phase is used!"

        lane_scores, traj_scores = output
        lane_scores = lane_scores.cpu().detach().numpy()
        traj_scores = traj_scores.cpu().detach().numpy()

        batch_size = len(sample_batched['seq_ids'])
        for key in sample_batched:
            if isinstance(sample_batched[key], torch.Tensor):
                sample_batched[key] = sample_batched[key].cpu().detach().numpy()
        seq_ids = sample_batched['seq_ids']
        cls_idx = sample_batched['cls_start_end_idx']
        trajs_idx = sample_batched['trajs_start_end_idx']

        # NOT direclty manipulate on the original data
        agent_futs_xy = np.copy(sample_batched['agent_futs_xy'])
        for sample_id, cls_start_end in enumerate(cls_idx):
            for fut_id in range(trajs_idx[cls_start_end[0]][0], trajs_idx[cls_start_end[1]-1][1]):
                # transform back to the original coords
                agent_futs_xy[fut_id, :, :] = agent_futs_xy[fut_id, :, :].dot(sample_batched['rotations'][sample_id].transpose()) \
                                              * sample_batched['scales'][sample_id] + sample_batched['translations'][[sample_id], :]

        if generate_traj:   # output trajectory prediction
            processed_output = {}
            nofilter_output = {}

            for sample_id in range(batch_size):
                cls_first, cls_last = cls_idx[sample_id]
                sample_lane_prob = softmax(lane_scores[cls_first:cls_last], axis=0).squeeze(1)
                sample_lane_guess_num = assign_guess_num(sample_lane_prob, num=_MAX_GUESSES_NUM)
                lane_assignment_order = sample_lane_prob.argsort()[::-1]
                filter_trajs = np.empty([0, agent_futs_xy.shape[1], 2])
                nofilter_trajs = np.empty([0, agent_futs_xy.shape[1], 2])
                for id in lane_assignment_order:
                    guess_num_this_lane = sample_lane_guess_num[id]
                    if guess_num_this_lane <= 0:
                        break
                    lane_id = cls_first + id
                    lane_futs_scores = traj_scores[trajs_idx[lane_id][0]:trajs_idx[lane_id][1]].squeeze(1)
                    lane_futs = agent_futs_xy[trajs_idx[lane_id][0]:trajs_idx[lane_id][1]]
                    lane_futs_ordered = lane_futs[lane_futs_scores.argsort()[::-1]]
                    nofilter_trajs = np.vstack( (nofilter_trajs, lane_futs_ordered[:guess_num_this_lane]) )
                    filter_trajs = self.preds_filter.add_predictions_by_least_dist(lane_futs_ordered, guess_num_this_lane, filter_trajs)
                if phase=='train':
                    processed_output[seq_ids[sample_id]] = nofilter_trajs
                elif phase=='validate':
                    processed_output[seq_ids[sample_id]] = filter_trajs
                    nofilter_output[seq_ids[sample_id]] = nofilter_trajs
                elif phase=='submit':
                    processed_output[seq_ids[sample_id]] = filter_trajs

            if phase == 'validate':
                return processed_output, nofilter_output, sample_batched
            else:
                return processed_output, sample_batched

        else:
            return output, sample_batched


def assign_guess_num(probability: np.ndarray, num:int):
    assignment = [0] * probability.size
    order = probability.argsort()[::-1]
    left_num = num
    for idx in order:
        assign_num = int(round(probability[idx] * num))
        left_num -= assign_num
        assignment[idx] = assign_num
        if left_num<=0:
            break
    if left_num > 0:
        assignment[order[0]] += left_num  # Increase the guess number of the high probable lane
    elif left_num < 0:
        assignment[idx] += left_num       # Reduce the guess number of the low probable lane
    return assignment


# Temporarily deprecated
def get_cond_probability(cls_start_end_idx: np.ndarray,
                         trajs_start_end_idx: np.ndarray,
                         lane_scores: np.ndarray,
                         traj_scores: np.ndarray):
    """
    Calculating the conditional probability for all the trajectories.
    The produced prob has the same shape with network output
    """
    prob = np.zeros_like(traj_scores)
    for sample_id, cls_start_end in enumerate(cls_start_end_idx):
        cls_first, cls_last = cls_start_end
        traj_first, traj_last = trajs_start_end_idx[cls_first][0], trajs_start_end_idx[cls_last - 1][1]
        sample_lane_prob = softmax(lane_scores[cls_first:cls_last], axis=0)
        for cls_id in range(cls_first, cls_last):
            lane_traj_first, lane_traj_last = trajs_start_end_idx[cls_id]
            lane_traj_prob = softmax(traj_scores[lane_traj_first:lane_traj_last], axis=0)
            prob[lane_traj_first:lane_traj_last] = lane_traj_prob * sample_lane_prob[cls_id - cls_first]
        # Normalize to probability 1 fore each sample
        prob[traj_first:traj_last] = prob[traj_first:traj_last] / prob[traj_first:traj_last].sum()
    return prob
