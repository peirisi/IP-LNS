import os
import torch
import torch_geometric


class GraphDataset(torch_geometric.data.Dataset):
    """
    This class encodes a collection of graphs, as well as a method to load such graphs from the disk.
    It can be used in turn by the data loaders provided by pytorch geometric.
    """

    def __init__(self, path):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.files = self._read_file(path)
        self.path = path

    def len(self):
        return len(self.files)

    def _read_file(self,path):
        files = os.listdir(path)
        pt_files = [f for f in files if f.endswith('.pt')]
        return pt_files
   
    def get(self, index):
        """
        This method loads a node bipartite graph observation as saved on the disk during data collection.
        """
        file = self.files[index]
        data = torch.load(self.path+file)

        graph = data['graph']
        
        sum=1
        for v in ['u','p','s','d','sc']:
            if 'x' in graph[v]:
                sum+=graph[v].num_nodes
            else:
                graph[v].num_nodes = 0
        for c in ['Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost','state_variable']:
            if 'x' in graph[c]:
                sum+=graph[c].num_nodes
            else:
                graph[c].num_nodes = 0
        graph['obj'].x = graph['obj'].x.reshape(1,-1)
        graph.num_nodes = sum
        
        graph.positive = data['positive']
        graph.negative = data['negative']
        return graph
    

