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
    BasicConv1d,
    BasicLinear,
    LaneEncoder,
    TrajEncoder,
    ScoreDecoder,
    CoreAttention,
    MultiHeadAttention,
)


###################################################################################
########################### Architecture ##########################################
###################################################################################
class TrajClassifier(nn.Module):
    def __init__(self, args):
        super(TrajClassifier, self).__init__()
        self.use_cuda = args.use_cuda
        self.loss_name = args.loss_name
        self.norm = 'GN'
        self.act = 'leaky_relu'
        self.obs_padding = True

        logging.info("<Core4(L2H, H2H, L2F, F2F)-sd2-L1>")
        self.preds_filter = PreditionsFilter(2.7, 0.0)

        # Parameters
        self.lane_embed_size = 16
        self.lane_enc_size = 64

        self.obs_embed_size = 32
        self.pred_embed_size = 32
        self.traj_enc_size = 128

        self.map_actor_key_size = 64
        self.actor_actor_key_size = 128

        self.combin_size = self.traj_enc_size * 3

        ## Feature encoding
        self.cls_encoder = LaneEncoder(self.lane_embed_size, self.lane_enc_size, norm=self.norm, act=self.act, bidirectional=True)
        self.obs_encoder_xy = TrajEncoder(self.obs_embed_size, self.traj_enc_size, norm=self.norm, act=self.act)
        self.obs_encoder_sd = TrajEncoder(self.obs_embed_size, self.traj_enc_size, norm=self.norm, act=self.act)
        self.nbrs_encoder_xy = TrajEncoder(self.obs_embed_size, self.traj_enc_size, norm=self.norm, act=self.act, dim_in=3 if self.obs_padding else 2)
        self.nbrs_encoder_sd = TrajEncoder(self.obs_embed_size, self.traj_enc_size, norm=self.norm, act=self.act)
        self.pred_encoder_xy = TrajEncoder(self.pred_embed_size, self.traj_enc_size, norm=self.norm, act=self.act)
        self.pred_encoder_sd = TrajEncoder(self.pred_embed_size, self.traj_enc_size, norm=self.norm, act=self.act)

        # Mapping feature to query / keys
        self.cls_feature_mapping = BasicLinear(self.lane_enc_size, self.map_actor_key_size, act=self.act)
        self.obs_feature_encoding = BasicLinear(2*self.traj_enc_size, self.traj_enc_size, act=self.act)
        self.obs_feature_mapping = BasicLinear(self.traj_enc_size, self.map_actor_key_size, act=self.act)
        self.nbrs_feature_encoding = BasicLinear(2*self.traj_enc_size, self.traj_enc_size, act=self.act)
        self.nbrs_feature_mapping = BasicLinear(self.traj_enc_size, self.map_actor_key_size, act=self.act)
        self.pred_feature_encoding = BasicLinear(2*self.traj_enc_size, self.traj_enc_size, act=self.act)
        self.pred_feature_mapping = BasicLinear(self.traj_enc_size, self.map_actor_key_size, act=self.act)

        # Mapping after attention
        self.social_feature_mapping = BasicLinear(self.traj_enc_size, self.actor_actor_key_size, act=self.act)
        self.allpreds_feature_mapping = BasicLinear(self.traj_enc_size, self.actor_actor_key_size, act=self.act)

        # Core attention funcs
        self.l2h_attention = CoreAttention(self.traj_enc_size, self.map_actor_key_size, self.lane_enc_size, act=self.act)
        self.h2h_attention = CoreAttention(self.traj_enc_size, self.actor_actor_key_size, self.traj_enc_size, act=self.act)
        self.l2f_attention = CoreAttention(self.traj_enc_size, self.map_actor_key_size, self.lane_enc_size, act=self.act)
        self.f2f_attention = CoreAttention(self.traj_enc_size, self.actor_actor_key_size, self.traj_enc_size, act=self.act)

        self.score_decoder = ScoreDecoder(n_in=self.combin_size, n_outs=[self.combin_size, self.combin_size, 1], act=self.act, dropout=0.1)


    def forward(self, sample_batched):
        device = torch.device("cuda:0")

        batch_size = len(sample_batched['seq_ids'])
        agent_cls, agent_obs_xy, agent_futs_xy = \
            sample_batched['agent_cls'], sample_batched['agent_obs_xy'], sample_batched['agent_futs_xy']
        nbrs_obs_xy, nbrs_obs_padding, nbrs_start_end_idx = \
            sample_batched['nbrs_obs_xy'], sample_batched['nbrs_obs_padding'], sample_batched['nbrs_start_end_idx']
        cl_start_end_idx , trajs_start_end_idx = \
            sample_batched['cls_start_end_idx'], sample_batched['trajs_start_end_idx']
        agent_obs_sds, agent_futs_sd, nbrs_obs_sds = \
            sample_batched['agent_obs_sds'], sample_batched['agent_futs_sd'], sample_batched['nbrs_obs_sds']

        ## Preprocess before feeding into-------------------------------------------------------------------
        cls_count_under_sample = [start_end[1] - start_end[0] for start_end in cl_start_end_idx]
        nbrs_count_under_sample = [start_end[1] - start_end[0] for start_end in nbrs_start_end_idx]
        preds_count_under_lane = [start_end[1] - start_end[0] for start_end in trajs_start_end_idx]
        preds_count_under_sample = [sum(preds_count_under_lane[start_end[0]:start_end[1]]) for start_end in cl_start_end_idx]

        # To GPU
        agent_cls = agent_cls.to(device)
        agent_obs_xy = agent_obs_xy.to(device)
        agent_futs_xy = agent_futs_xy.to(device)
        nbrs_obs_xy = nbrs_obs_xy.to(device)
        nbrs_obs_padding = nbrs_obs_padding.to(device)
        # nbrs_start_end_idx = nbrs_start_end_idx.to(device)
        # cl_start_end_idx = cl_start_end_idx.to(device)
        # trajs_start_end_idx = trajs_start_end_idx.to(device)
        agent_obs_sds = agent_obs_sds.to(device)
        agent_futs_sd = agent_futs_sd.to(device)
        nbrs_obs_sds = nbrs_obs_sds.to(device)

        ## Global Propagation -----------------------------------------------------------------------------
        cls_encoding, cls_state_encoding = self.cls_encoder(agent_cls)            # total_cls x num_parts x lane_enc_size / total_cls x lane_enc_size
        cls_mapping = self.cls_feature_mapping(cls_encoding)

        obs_encoding_xy = self.obs_encoder_xy(agent_obs_xy)
        obs_encoding_sds = self.obs_encoder_sd(agent_obs_sds)
        obs_encoding = torch.cat([torch.cat([obs_encoding_xy[[i]].repeat(cls_count, 1) for i, cls_count in enumerate(cls_count_under_sample)],dim=0),
                                  obs_encoding_sds], dim=-1)                          # num_samples x traj_enc_size
        obs_encoding = self.obs_feature_encoding(obs_encoding)
        obs_mapping = self.obs_feature_mapping(obs_encoding)


        nbrs_encoding_xy = self.nbrs_encoder_xy(torch.cat([nbrs_obs_xy, nbrs_obs_padding.unsqueeze(-1)], dim=-1)) if self.obs_padding else self.nbrs_encoder_xy(nbrs_obs_xy)
        nbrs_encoding_sds = self.nbrs_encoder_sd(nbrs_obs_sds)
        nbrs_encoding = torch.cat([torch.cat([nbrs_encoding_xy[nbrs_start_end[0]:nbrs_start_end[1]].repeat([cls_count, 1]) for nbrs_start_end, cls_count in zip(nbrs_start_end_idx, cls_count_under_sample)], dim=0),
                                   nbrs_encoding_sds], dim=-1)
        nbrs_encoding = self.nbrs_feature_encoding(nbrs_encoding)
        nbrs_mapping = self.nbrs_feature_mapping(nbrs_encoding)
        nbrs_encoding_start_id_under_sample = np.cumsum([0] + [i*j for i,j in zip(nbrs_count_under_sample, cls_count_under_sample)])


        pred_encoding_xy = self.pred_encoder_xy(agent_futs_xy)
        pred_encoding_sds = self.pred_encoder_sd(agent_futs_sd)
        pred_encoding = torch.cat([pred_encoding_xy,
                                   pred_encoding_sds], dim=-1)  # total_preds x traj_enc_size
        pred_encoding = self.pred_feature_encoding(pred_encoding)
        pred_mapping = self.pred_feature_mapping(pred_encoding)


        ## Batch loop ----------------------------------------------------------------------------------------------
        combin = torch.empty((0, self.combin_size)).cuda() if self.use_cuda else torch.empty((0, self.combin_size))
        for k in range(batch_size):
            cls_range = cl_start_end_idx[k]
            nbr_range = nbrs_start_end_idx[k]
            nbr_count = nbr_range[1] - nbr_range[0]

            ## Transform all encs to [batch(pred_num) x num_obj x feature_dim]
            obs_encs = torch.cat([obs_encoding[[cls_id]].repeat([preds_count_under_lane[cls_id], 1, 1]) for cls_id in range(cls_range[0], cls_range[1])], dim=0)
            obs_maps = torch.cat([obs_mapping[[cls_id]].repeat([preds_count_under_lane[cls_id], 1, 1]) for cls_id in range(cls_range[0], cls_range[1])], dim=0)

            nbr_encs = torch.cat([nbrs_encoding[nbrs_encoding_start_id_under_sample[k] + nbr_count*i :
                                                nbrs_encoding_start_id_under_sample[k] + nbr_count*(i+1)].unsqueeze(0).repeat([preds_count_under_lane[cls_id],1,1])
                                  for i, cls_id in enumerate(range(cls_range[0], cls_range[1]))], dim=0)
            nbr_maps = torch.cat([nbrs_mapping[nbrs_encoding_start_id_under_sample[k] + nbr_count*i :
                                               nbrs_encoding_start_id_under_sample[k] + nbr_count*(i+1)].unsqueeze(0).repeat([preds_count_under_lane[cls_id],1,1])
                                  for i, cls_id in enumerate(range(cls_range[0], cls_range[1]))], dim=0)

            pred_encs = pred_encoding[trajs_start_end_idx[cls_range[0]][0]:trajs_start_end_idx[cls_range[1] - 1][1]].unsqueeze(1)
            pred_maps = pred_mapping[trajs_start_end_idx[cls_range[0]][0]:trajs_start_end_idx[cls_range[1] - 1][1]].unsqueeze(1)

            cls_encs = torch.cat([cls_encoding[cls_id].unsqueeze(0).repeat(preds_count_under_lane[cls_id], 1, 1) for cls_id in range(cls_range[0], cls_range[1])], dim=0)
            cls_maps = torch.cat([cls_mapping[cls_id].unsqueeze(0).repeat(preds_count_under_lane[cls_id], 1, 1) for cls_id in range(cls_range[0], cls_range[1])], dim=0)


            ## Lane-AllObs (L2H) attention:
            allobs_encs = torch.cat((nbr_encs, obs_encs), dim=1)
            allobs_maps = torch.cat((nbr_maps, obs_maps), dim=1)
            allobs_lane_output = self.l2h_attention(allobs_encs, allobs_maps, cls_maps, cls_encs)

            ## Social (H2H) attention
            allobs_lane_output_map = self.social_feature_mapping(allobs_lane_output)
            social_output = self.h2h_attention(allobs_lane_output, allobs_lane_output_map, allobs_lane_output_map, allobs_lane_output)

            ## Lane-Futs (L2F) attention:
            pred_lane_output = self.l2f_attention(pred_encs, pred_maps, cls_maps, cls_encs)

            ## Futs-Futs (F2F) attention:
            pred_lane_output = pred_lane_output.transpose(0, 1)
            pred_lane_output_map = self.allpreds_feature_mapping(pred_lane_output)
            allpreds_output = self.f2f_attention(pred_lane_output,
                                                 pred_lane_output_map, pred_lane_output_map, pred_lane_output)

            ## Merge together
            agent_obs_part = allobs_lane_output[:, [-1], :]
            agent_interaction_part = social_output[:, [-1], :]
            agent_pred_part = allpreds_output.transpose(0,1)
            under_this_sample = torch.cat((agent_obs_part, agent_interaction_part, agent_pred_part), dim=-1)

            combin = torch.cat((combin, under_this_sample.squeeze(1)), dim=0)

        output = self.score_decoder(combin)  # num_predictions x 1
        return output

    def post_process(self, dataset, output, sample_batched,
                     generate_traj: bool = True,
                     phase: str = 'train'):
        """
        Network output may be further processed, and take the full sample_batched back to CPU
        phase: here it corresponds to the function of train() validate() submit() in predictor file.
        """
        assert phase in ['train', 'validate', 'submit'], "Wrong type of argu phase is used!"

        output = output.cpu().detach().numpy()
        for key in sample_batched:
            if isinstance(sample_batched[key], torch.Tensor):
                sample_batched[key] = sample_batched[key].cpu().detach().numpy()
        seq_ids = sample_batched['seq_ids']
        cls_idx = sample_batched['cls_start_end_idx']
        trajs_idx = sample_batched['trajs_start_end_idx']

        # NOT direclty manipulate on the original data
        agent_futs_xy = np.copy(sample_batched['agent_futs_xy'])
        for i, start_end in enumerate(cls_idx):
            for fut_id in range(trajs_idx[start_end[0]][0], trajs_idx[start_end[1]-1][1]):
                # transform back to the original coords
                agent_futs_xy[fut_id, :, :] = agent_futs_xy[fut_id, :, :].dot(sample_batched['rotations'][i].transpose()) \
                                              * sample_batched['scales'][i] + sample_batched['translations'][[i], :]

        if generate_traj:   # output trajectory prediction
            processed_output = {}
            processed_probability = {}
            nofilter_output = {}
            for i, start_end in enumerate(cls_idx):
                first = trajs_idx[start_end[0]][0]
                last = trajs_idx[start_end[1] - 1][1]
                sample_output = output[first:last].squeeze(1)
                ordered_indices = self.get_output_ordered_indices(sample_output)

                sample_futs = agent_futs_xy[first:last]
                sample_futs_ordered = sample_futs[ordered_indices]

                assert self.loss_name.lower() != "pred_regress"     # Only works with scoring type of output
                sample_probs = softmax(sample_output, axis=0)
                sample_probs_ordered = sample_probs[ordered_indices]

                if phase=='train':
                    # No filter would be used
                    processed_output[seq_ids[i]] = sample_futs_ordered[:_MAX_GUESSES_NUM]
                elif phase=='validate':
                    # Return two types of processed output
                    processed_output[seq_ids[i]], indices = self.preds_filter.add_predictions_by_least_dist(sample_futs_ordered, num=_MAX_GUESSES_NUM)
                    sample_p = sample_probs_ordered[indices]
                    processed_probability[seq_ids[i]] = sample_p / sample_p.sum()

                    nofilter_output[seq_ids[i]] = sample_futs_ordered[:_MAX_GUESSES_NUM]
                elif phase=='submit':
                    # Only the filtered results is used
                    processed_output[seq_ids[i]], indices = self.preds_filter.add_predictions_by_least_dist(sample_futs_ordered, num=_MAX_GUESSES_NUM)
                    sample_p = sample_probs_ordered[indices]
                    processed_probability[seq_ids[i]] = sample_p / sample_p.sum()

            if phase=='train':
                return processed_output, sample_batched
            elif phase == 'validate':
                return processed_output, processed_probability, nofilter_output, sample_batched
            elif phase == 'submit':
                return processed_output, processed_probability

        else:
            return output, sample_batched


    def get_output_ordered_indices(self, sample_output):
        if self.loss_name.lower() == "pred_regress":
            # Output disp error wrt ground truth
            return sample_output.argsort()
        else:
            # Output scores
            return sample_output.argsort()[::-1]


# Used for score/probability-based output (NOT error-based)
def get_probability(cls_start_end_idx: np.ndarray,
                    trajs_start_end_idx: np.ndarray,
                    traj_scores: np.ndarray):
    """
    Calculating the probability for all the trajectories.
    The produced prob has the same shape with network output
    """
    prob = np.zeros_like(traj_scores)
    for sample_id, cls_start_end in enumerate(cls_start_end_idx):
        cls_first, cls_last = cls_start_end
        traj_first = trajs_start_end_idx[cls_first][0]
        traj_last = trajs_start_end_idx[cls_last -1][1]
        prob[traj_first:traj_last] = softmax(traj_scores[traj_first:traj_last], axis=0)
    return prob.flatten()
