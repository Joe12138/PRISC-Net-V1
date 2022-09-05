import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionFCLayer(nn.Module):
    """
    Self-attention layer. No scale_factor d_k
    """
    def __init__(self,
                 in_channels: int,
                 global_graph_width: int,
                 need_scale: bool):
        super(SelfAttentionFCLayer, self).__init__()

        self.in_channels = in_channels
        self.graph_width = global_graph_width

        self.q_lin = nn.Linear(in_channels, global_graph_width)
        self.k_lin = nn.Linear(in_channels, global_graph_width)
        self.v_lin = nn.Linear(in_channels, global_graph_width)

        self.scale_factor_d = 1 + int(np.sqrt(self.in_channels)) if need_scale else 1

    def forward(self, x, valid_lens):
        query = self.q_lin(x)
        key = self.k_lin(x)
        value = self.v_lin(x)

        scores = torch.bmm(query, key.transpose(1, 2))/math.sqrt(self.graph_width)
        attention_weights = self.masked_softmax(scores, valid_lens)
        x = torch.bmm(attention_weights, value)

        return x

    @staticmethod
    def masked_softmax(x, valid_lens):
        """
        masked softmax for attention scores
        :param x: 3-D tensor
        :param valid_lens: 1-D or 2-D tensor
        :return:
        """
        if valid_lens is None:
            return F.softmax(x, dim=1)
        else:
            shape = x.shape
            if valid_lens.shape[0] != shape[0]:
                valid_len = torch.repeat_interleave(valid_lens, repeats=shape[0], dim=0)
            else:
                valid_len = valid_lens.reshape(-1)

            # Fill masked element with a large negative, whose exp is 0.
            mask = torch.zeros_like(x, dtype=torch.bool)
            for batch_id, cnt in enumerate(valid_len):
                cnt = int(cnt.detach().cpu().numpy())
                mask[batch_id, :, cnt:] = True
                mask[batch_id, cnt:] = True

            x_masked = x.masked_fill(mask, -1e12)
            return F.softmax(x_masked, dim=-1)*(1-mask.float())


class GlobalGraph(nn.Module):
    """
    Global graph that compute the global information.
    """
    def __init__(self,
                 in_channels: int,
                 global_graph_width: int,
                 num_global_layers: int,
                 need_scale: bool):
        super(GlobalGraph, self).__init__()
        self.in_channels = in_channels # 66
        self.global_graph_width = global_graph_width # 64

        self.layers = nn.Sequential()

        for i in range(num_global_layers):
            self.layers.add_module(
                f'glp_{i}',
                SelfAttentionFCLayer(in_channels,
                                     self.global_graph_width,
                                     need_scale)
            )
            in_channels = self.global_graph_width

    def forward(self, x, **kwargs):
        for _, layer in self.layers.named_modules():
            if isinstance(layer, SelfAttentionFCLayer):
                x = layer(x, **kwargs)

        return x
