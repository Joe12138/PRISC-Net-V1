import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import max_pool

from target_prediction.model.layers.basic_module import MLP


class SubGraph(nn.Module):
    """
    Multi-layer MLP
    Subgraph that computes all vectors in a polyline, and get a polyline-level feature
    """
    def __init__(self,
                 in_channels: int,
                 num_subgraph_layers: int,
                 hidden_unit: int):
        super(SubGraph, self).__init__()

        self.num_subgraph_layers = num_subgraph_layers # 3
        self.hidden_unit = hidden_unit # 64
        self.out_channels = hidden_unit # 64

        self.layer_seq = nn.Sequential()

        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(
                f'glp_{i}',
                MLP(
                    in_channel=in_channels,
                    out_channel=hidden_unit,
                    hidden=hidden_unit
                )
            )
            in_channels = hidden_unit * 2

        self.linear = nn.Linear(hidden_unit*2, hidden_unit)

    def forward(self, sub_data):
        """
        Polyline vector set in torch_geometric.data.Data format
        :param sub_data: (Data) [x, y, cluster, edge_index, valid_len]
        :return:
        """
        x = sub_data.x
        sub_data.cluster = sub_data.cluster.long()
        sub_data.edge_index = sub_data.edge_index.long()

        for _, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                sub_data.x = x
                agg_data = max_pool(sub_data.cluster, sub_data)

                x = torch.cat([sub_data.x, agg_data.x[sub_data.cluster]], dim=1)

        x = self.linear(x)
        sub_data.x = x
        out_data = max_pool(sub_data.cluster, sub_data)
        x = out_data.x

        assert x.shape[0] % int(sub_data.time_step_len[0]) == 0

        return F.normalize(x, p=2.0, dim=1)  # L2 normalization
