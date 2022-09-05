from torch_geometric.data import Data

class GraphData(Data):
    """
    Override key 'cluster' indicating which polyline_id is for the vector.
    """
    def __inc__(self, key, value):
        if key == "edge_index":
            return self.x.size(0)
        elif key == "cluster":
            return int(self.cluster.max().item())+1
        else:
            return 0