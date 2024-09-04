import copy
import torch
import torch_geometric

class BipartiteGraphConvolution(torch_geometric.nn.MessagePassing):
    """
    The bipartite graph convolution is already provided by pytorch geometric and we merely need
    to provide the exact form of the messages being passed.
    """

    def __init__(self,emb_size):
        super().__init__("add")
            
        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size)
        )
        self.feature_module_edge = torch.nn.Sequential(
            torch.nn.Linear(2, emb_size, bias=False)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(emb_size, emb_size, bias=False)
        )
        self.feature_module_final = torch.nn.Sequential(
            torch.nn.LayerNorm(emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )
        
    

    def forward(self, left_features, edge_indices, edge_features, right_features):
        output = self.propagate(
            edge_indices,
            size=(left_features.shape[0], right_features.shape[0]),
            node_features=(left_features, right_features),
            edge_features=edge_features,
        )
        return output
            
    def message(self, node_features_i, node_features_j, edge_features):
        output = self.feature_module_final(
            self.feature_module_left(node_features_i)
            + self.feature_module_edge(edge_features)
            + self.feature_module_right(node_features_j)
        )
        return output
    


class GNNPolicy(torch.nn.Module):
    def __init__(self):
        super(GNNPolicy, self).__init__()
        self.emb_size=64
        self.var_node = ['u','p','s','d','sc']
        self.con_node = ['Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost','state_variable']

        self.obj_node = ['obj']
        # self.node_types=['u','p','s','Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost','obj']
        self.relations = ['v2o','o2c','v2c','c2o','o2v','c2v']
        
        # self.init_sizes=[7,7,7,6,6,6,6,6,6,6,2]
        edge_feats=2
        # self.num_relations=len(self.var_node)*len(self.con_node)
        self.vc_idx={}
        self.oc_idx={}
        self.ov_idx={}
        # self.node_types = ['u','p','s','Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost']    
        self.node_embedding = torch.nn.ModuleList()
        
        for i in [14,6,2]:
            lin = torch.nn.Sequential(
                torch.nn.LayerNorm(i),
                torch.nn.Linear(i, self.emb_size),
                torch.nn.ReLU(),
                torch.nn.Linear(self.emb_size,self.emb_size),
                torch.nn.ReLU()
            )
            self.node_embedding.append(lin)
        self.edge_embedding = torch.nn.LayerNorm(edge_feats)


        self.con1=torch.nn.Sequential()
        self.emb1=torch.nn.Sequential()
        self.con2=torch.nn.Sequential()
        self.emb2=torch.nn.Sequential()
        self.conidx = {}
        self.embidx = {}
        cntc=0
        cnte=0
        all_nodes = self.var_node+self.con_node+self.obj_node
        for ori in [self.var_node, self.con_node, self.obj_node]:
            tar = [node for node in all_nodes if node is not ori]
            for node in ori:
                for node2 in tar:
                    self.con1.append(BipartiteGraphConvolution(self.emb_size))
                    self.con2.append(BipartiteGraphConvolution(self.emb_size))
                    self.conidx[(node,node2)]=cntc
                    cntc+=1
                self.emb1.append(self.down_scale(self.emb_size))
                self.emb2.append(self.down_scale(self.emb_size))
                self.embidx[node]=cnte
                cnte+=1

        self.ln = torch.nn.LayerNorm(self.emb_size)


        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(self.emb_size, self.emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(self.emb_size, 1, bias=False)
        )
    def down_scale(self,emb_size):
        return torch.nn.Sequential(
            torch.nn.Linear(2 * emb_size, emb_size),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_size, emb_size),
        )

        
    def trans_dimensions(self, g):
        data = copy.deepcopy(g)
        self.con_node = ['Minimum_up/down_time_constraints','Unit_generation_limits','Power_balance_constrains','System_spinning_reserve_requirement','Ramp_rate_limits','Initial_status_of_units','startup_cost','state_variable']
        self.con_node = [c for c in self.con_node if 'x' in data[c]]
        self.var_node = ['u','p','s','d','sc']
        self.var_node = [v for v in self.var_node if 'x' in data[v]]

        for v in self.var_node:
            data[v].x = self.node_embedding[0](data[v].x)

        for c in self.con_node:
            data[c].x = self.node_embedding[1](data[c].x)

        data['obj'].x = self.node_embedding[2](data['obj'].x)
        cnt=0
        for v in self.var_node:
            for c in self.con_node:
                if 'edge_attr' in data[v,'v2c',c]:
                    data[v,'v2c',c].edge_attr = self.edge_embedding(data[v,'v2c',c].edge_attr)
                    data[c,'c2v',v].edge_attr = data[v,'v2c',c].edge_attr
                    self.vc_idx[(v,c)]=cnt
                    self.vc_idx[(c,v)]=cnt
                else:
                    self.vc_idx[(v,c)]=-1
                    self.vc_idx[(c,v)]=-1
                cnt+=1
        cnt=0
        for c in self.con_node:
            if 'edge_attr' in data['obj','o2c',c]:
                data['obj','o2c',c].edge_attr = self.edge_embedding(data['obj','o2c',c].edge_attr)
                data[c,'c2o','obj'].edge_attr = data['obj','o2c',c].edge_attr
                self.oc_idx[('obj',c)]=cnt
                self.oc_idx[(c,'obj')]=cnt
            else:
                self.oc_idx[('obj',c)]=-1
                self.oc_idx[(c,'obj')]=-1
            cnt+=1
        cnt=0
        for v in self.var_node:
            if 'edge_attr' in data['obj','o2v',v]:
                data['obj','o2v',v].edge_attr = self.edge_embedding(data['obj','o2v',v].edge_attr)
                data[v,'v2o','obj'].edge_attr = data['obj','o2v',v].edge_attr
                self.ov_idx[('obj',v)]=cnt
                self.ov_idx[(v,'obj')]=cnt
            else:
                self.ov_idx[('obj',v)]=-1
                self.ov_idx[(v,'obj')]=-1
            cnt+=1

        self.target_nodes = [self.obj_node,self.con_node,self.con_node,self.obj_node,self.var_node,self.var_node]
        self.source_nodes = [self.var_node,self.obj_node,self.var_node,self.con_node,self.obj_node,self.con_node]
        return data
    def process_nodes(self, target_node, source_node, emb, conv, data, edge_type):
        if 'o' in edge_type and 'v' in edge_type:
            idx = self.ov_idx
        elif 'o' in edge_type and 'c' in edge_type:
            idx = self.oc_idx
        else:
            idx = self.vc_idx
        for node in target_node:
            x = torch.zeros_like(data[node].x)
            for other_node in source_node:
                if idx[(other_node, node)] != -1:
                    
                    x += conv[self.conidx[(other_node,node)]](
                        data[other_node].x, 
                        data[other_node, edge_type, node].edge_index,
                        data[other_node, edge_type, node].edge_attr,
                        data[node].x
                    )
            data[node].x = emb[self.embidx[node]](torch.cat([data[node].x, self.ln(x)], dim=-1))

    
    def forward(self, data):
        data = self.trans_dimensions(data)
        for i,(tar,sou,rel) in enumerate(zip(self.target_nodes,self.source_nodes,self.relations)):
            self.process_nodes(tar, sou, self.emb1, self.con1, data, rel)

        
        for i,(tar,sou,rel) in enumerate(zip(self.target_nodes,self.source_nodes,self.relations)):
            self.process_nodes(tar, sou, self.emb2, self.con2, data, rel)


        x = self.output_module(data['u'].x).sigmoid()
        return x

