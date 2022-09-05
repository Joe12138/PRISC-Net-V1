import torch.nn as nn
import torch

from target_prediction.model.layers.subgraph import SubGraph
from target_prediction.model.layers.global_graph import GlobalGraph
from target_prediction.model.layers.basic_module import MLP


class VectorNetBackbone(nn.Module):
    """
    Hierarchical GNN with trajectory prediction MLP
    """
    def __init__(self,
                 in_channels: int,
                 num_subgraph_layers: int,
                 num_global_graph_layer: int,
                 subgraph_width: int,
                 global_graph_width: int,
                 aux_mlp_width: int,
                 with_aux: bool,
                 device: torch.device("cuda")):
        super(VectorNetBackbone, self).__init__()

        self.num_subgraph_layers = num_subgraph_layers
        self.global_graph_width = global_graph_width

        self.device = device

        self.subgraph = SubGraph(in_channels=in_channels,
                                 num_subgraph_layers=num_subgraph_layers,
                                 hidden_unit=subgraph_width)

        self.global_graph = GlobalGraph(in_channels=self.subgraph.out_channels+2,
                                        global_graph_width=self.global_graph_width,
                                        num_global_layers=num_global_graph_layer,
                                        need_scale=False)

        # Auxiliary recovery MLP
        self.with_aux = with_aux
        if self.with_aux:
            self.aux_mlp = nn.Sequential(
                MLP(in_channel=self.global_graph_width,
                    out_channel=aux_mlp_width,
                    hidden=aux_mlp_width),
                nn.Linear(aux_mlp_width, self.subgraph.out_channels)
            )

    def forward(self, data, mode="normal"):
        """
        :param data: (Data) [x, y, cluster, edge_index, valid_len]
        :return:
        """
        if mode == "normal":
            batch_size = data.num_graphs
        else:
            batch_size = 1
        time_step_len = data.time_step_len[0].int()
        valid_lens = data.valid_len

        id_embedding = data.identifier
        sub_graph_out = self.subgraph(data)

        if self.training and self.with_aux:
            randoms = 1 + torch.rand((batch_size,), device=self.device) * (valid_lens - 2) + \
                      time_step_len * torch.arange(batch_size, device=self.device)
            # mask_polyline_indices = [torch.randint(1, valid_lens[i] - 1) + i * time_step_len for i in range(batch_size)]
            mask_polyline_indices = randoms.long()
            aux_gt = sub_graph_out[mask_polyline_indices]
            sub_graph_out[mask_polyline_indices] = 0.0

            # reconstruct the batch global interaction graph data
        x = torch.cat([sub_graph_out, id_embedding], dim=1).view(batch_size, -1, self.subgraph.out_channels + 2)
        valid_lens = data.valid_len

        if self.training:
            # mask out the features for a random subset of polyline nodes
            # for one batch, we mask the same polyline features

            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            if self.with_aux:
                aux_in = global_graph_out.view(-1, self.global_graph_width)[mask_polyline_indices]
                aux_out = self.aux_mlp(aux_in)

                return global_graph_out, aux_out, aux_gt

            return global_graph_out, None, None

        else:
            # global_graph_out = self.global_graph(sub_graph_out, batch_size=data.num_graphs)
            global_graph_out = self.global_graph(x, valid_lens=valid_lens)

            return global_graph_out, None, None