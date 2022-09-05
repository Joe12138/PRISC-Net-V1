import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List

###################################################################################
########################### Basic Part #####################################
###################################################################################
_LEAKY_SLOPE = 0.1  # slope for Leaky ReLU
_GN_GROUP = 4       # least number of groups for Group Normalization


class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, norm=None, act=None, stride=1, padding=0, dilation=1):
        super(BasicConv1d, self).__init__()
        self.apply_norm = False
        self.apply_act = False

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=(norm is None))
        if norm:
            self.apply_norm = True
            assert (norm in ['BN', 'LN', 'GN', 'IN'])
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(out_channels)
            elif norm == 'LN':
                self.norm = nn.GroupNorm(1, out_channels)
            elif norm == 'GN':
                self.norm = nn.GroupNorm(_GN_GROUP, out_channels)
            elif norm == 'IN':
                self.norm = nn.GroupNorm(out_channels, out_channels)
        if act:
            assert (act in ['relu', 'leaky_relu'])
            self.apply_act = True
            if act == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif act == 'leaky_relu':
                self.act = nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)

    def forward(self, x):
        # try:
        #     out = self.conv(x)
        # except RuntimeError:
        #     print("Wrong here!")
        out = self.conv(x)
        if self.apply_norm:
            out = self.norm(out)
        if self.apply_act:
            out = self.act(out)
        return out


class BasicLinear(nn.Module):
    def __init__(self, n_in, n_out, norm=None, act=None):
        super(BasicLinear, self).__init__()
        self.apply_norm = False
        self.apply_act = False

        self.linear = nn.Linear(n_in, n_out, bias=(norm is None))
        if norm:
            self.apply_norm = True
            assert (norm in ['BN', 'LN', 'GN', 'IN'])
            if norm == 'BN':
                self.norm = nn.BatchNorm1d(n_out)
            elif norm == 'LN':
                self.norm = nn.GroupNorm(1, n_out)
            elif norm == 'GN':
                self.norm = nn.GroupNorm(_GN_GROUP, n_out)
            elif norm == 'IN':
                self.norm = nn.GroupNorm(n_out, n_out)
        if act:
            assert (act in ['relu', 'leaky_relu'])
            self.apply_act = True
            if act == 'relu':
                self.act = nn.ReLU(inplace=True)
            elif act == 'leaky_relu':
                self.act = nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)

    def forward(self, x):
        out = self.linear(x)
        if self.apply_norm:
            if len(x.shape)==2:
                out = self.norm(out)
            elif len(x.shape)==3:       # The nomalized dim is the last one as it's previously processed by Linear
                out = self.norm(out.permute(0, 2, 1)).permute(0, 2, 1)
            else:
                assert False, "Unsupported input size"
        if self.apply_act:
            out = self.act(out)
        return out

#############################################################################
########################### Lane Module #####################################
#############################################################################

class LaneEncoderConv(nn.Module):

    def __init__(self, enc_size, norm, act):
        super(LaneEncoderConv, self).__init__()
        self.conv0 = BasicConv1d(in_channels=2, out_channels=8, kernel_size=3, norm=norm, act=act)
        self.conv1 = BasicConv1d(in_channels=8, out_channels=32, kernel_size=3, norm=norm, act=act)
        self.conv2 = BasicConv1d(in_channels=32, out_channels=enc_size, kernel_size=3, norm=norm, act=act)
        self.pool1 = nn.AvgPool1d(kernel_size=3, stride=1)

    def forward(self, x):
        # x: 3-D tensor, num_centerlines x num_pts x 2
        x = x.permute(0, 2, 1)
        conv0_out = self.conv0(x)
        conv1_out = self.conv1(conv0_out)
        conv2_out = self.conv2(conv1_out)
        pool_out = self.pool1(conv2_out)            # 3D tensor: num_centerlines x lane_embed_size x (lane_segment_num - 2*n)
        return pool_out.permute(0, 2, 1)


class LaneEncoder(nn.Module):

    def __init__(self, embed_size, enc_size, norm, act, num_layers=1, bidirectional=True):
        super(LaneEncoder, self).__init__()
        self.enc_size = enc_size
        self.temp_conv = BasicConv1d(in_channels=2, out_channels=embed_size, kernel_size=3, norm=norm, act=act)
        hidden_size = enc_size//2 if bidirectional else enc_size
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, x):
        # x: 3-D tensor, num_centerlines x num_pts x 2
        x = x.permute(0, 2, 1)
        x = self.temp_conv(x)
        x = x.permute(2, 0, 1)
        output, (hidden_n, cell_n) = self.lstm(x) # output: 3D tensor, seq_len x num_cls (batch) x enc_size
        seq_states = output.permute(1,0,2)
        final_state = hidden_n.permute(1, 0, 2).contiguous().view(-1, self.enc_size)
        return seq_states, final_state

#############################################################################
########################### Trajecotry Module ###############################
#############################################################################

class TrajEncoder(nn.Module):

    def __init__(self, embed_size, enc_size, norm, act, dim_in=2, num_layers=1):
        super(TrajEncoder, self).__init__()
        self.temp_conv = BasicConv1d(in_channels=dim_in, out_channels=embed_size, kernel_size=3, norm=norm, act=act, padding=0)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=enc_size, num_layers=num_layers)

    def forward(self, x):
        # x: 3-D tensor, num_trajs x seq_len x 2
        x = x.permute(0, 2, 1)
        x = self.temp_conv(x)
        x = x.permute(2, 0, 1)
        _, (hidden_n, _) = self.lstm(x)
        return hidden_n.squeeze(0)                          # hidden_n: 3D tensor, 1 x num_trajs x enc_size


#############################################################################
########################### Classify Module ###############################
#############################################################################

class ScoreDecoder(nn.Module):

    def __init__(self, n_in, n_outs, act, dropout=0):
        super(ScoreDecoder, self).__init__()

        net = []
        n_outs = [n_in] + n_outs
        for i in range(len(n_outs)-2):
            net.append(BasicLinear(n_outs[i], n_outs[i+1], norm=None, act=act))
            if dropout>0:
                net.append(nn.Dropout(dropout))

        net.append(BasicLinear(n_outs[-2], n_outs[-1], norm=None, act=None))
        self.fc = nn.Sequential(*net)

    def forward(self, x):
        scoring = self.fc(x)
        return scoring


#############################################################################
########################### Attention Module ###############################
#############################################################################
class PositionalEmbedding(nn.Module):
    def __init__(self, d_emb):
        super(PositionalEmbedding, self).__init__()

        self.d_emb = d_emb

        inv_freq = 1/(10000**torch.arange(0.0, d_emb, 2.0)/d_emb)

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)

        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


#############################################################################
########################### Attention Module ###############################
#############################################################################


class CoreAttention(nn.Module):

    def __init__(self, dim_model, dim_key, dim_val, act, fc_layers=1, dropout=0):
        super(CoreAttention, self).__init__()
        assert (act in ['relu', 'leaky_relu'])
        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_val = dim_val

        # self.pos_emb = PositionalEmbedding(self.dim_model)

        self.apply_dp = dropout > 0
        if self.apply_dp:
            self.attn_dropout = nn.Dropout(dropout)
            self.fc_dropout = nn.Dropout(dropout)

        if fc_layers == 1:
            self.fc = BasicLinear(dim_val, dim_model, norm=None, act=None)
        else:
            fc = [BasicLinear(dim_val, dim_model, norm=None, act=act) for _ in range(fc_layers-1)] + \
                 [BasicLinear(dim_model, dim_model, norm=None, act=None)]
            self.fc = nn.Sequential(*fc)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)

    def forward(self, model_input, query, key, value):
        # repeat_num, num_vec = model_input.shape[0], model_input.shape[1]
        # pos_seq = torch.arange(num_vec-1, -1, -1.0, device=model_input.device, dtype=model_input.dtype)
        # pos_emb = self.pos_emb(pos_seq, repeat_num)
        attn = torch.matmul(query, key.transpose(1,2)) /  (self.dim_key ** 0.5)
        attn = F.softmax(attn, dim=-1)
        if self.apply_dp:
            attn = self.attn_dropout(attn)
        residual = torch.matmul(attn, value)
        residual = self.fc(residual)
        if self.apply_dp:
            residual = self.fc_dropout(residual)
        output = self.act(model_input + residual)
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, dim_q, dim_k, dim_key, dim_val, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_key = dim_key
        self.dim_val = dim_val

        self.w_qs = nn.Linear(dim_q, num_head * dim_key, bias=False)
        self.w_ks = nn.Linear(dim_k, num_head * dim_key, bias=False)
        self.w_vs = nn.Linear(dim_k, num_head * dim_key, bias=False)
        self.fc = nn.Linear(num_head * dim_val, dim_q, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.fc_dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_q, eps=1e-6)

    def scaled_dot_product_attention(self, q, k, v):
        attn = torch.matmul(q, k.transpose(2, 3)) / (self.dim_key ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        output = torch.matmul(attn, v)
        return output, attn

    def forward(self, q, k, v):
        residual = q
        num_head, dim_key, dim_val = self.num_head, self.dim_key, self.dim_val
        q_shape, k_shape, v_shape = q.shape, k.shape, v.shape
        # Separate different heads
        q = self.w_qs(q).view(q_shape[0], q_shape[1], num_head, dim_key)
        k = self.w_ks(k).view(k_shape[0], k_shape[1], num_head, dim_key)
        v = self.w_vs(v).view(v_shape[0], v_shape[1], num_head, dim_val)
        # Transpose for attention dot product
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Transpose to move the head dimension back to [batch x num_q x num_head x dim_val]
        # Combine the last two dimensions to concatenate all the heads together to [batch x num_q x (num_head * dim_val) ]
        output, attn = self.scaled_dot_product_attention(q, k, v)
        output = output.transpose(1, 2).contiguous().view(q_shape[0], q_shape[1], -1)
        output = self.fc_dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn