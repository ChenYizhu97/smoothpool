#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
@author: Yizhu Chen
"""

import torch
from sparsemax import Sparsemax
from torch.nn import Sigmoid, BatchNorm1d, ReLU, Tanh
from torch.tensor import Tensor
from torch_geometric.nn import MessagePassing
from torch.nn import Linear
from utils.utils import dense_to_sparse, dense_to_sparse_with_attr
from torch_geometric.utils import num_nodes, to_dense_adj, to_dense_batch
from torch_geometric.typing import Tensor

def smoothness_attention(x: Tensor, edge_index: Tensor, smoothness_gate: Tensor):
    source, target = edge_index
    #print(target, smoothness_gate)
    N = num_nodes.maybe_num_nodes(target, None)
    out = torch.zeros((N,), dtype=x.dtype, device=x.device)
    out = out.scatter_add_(0, target, smoothness_gate)
    attention_score =  smoothness_gate/out[target]
    return attention_score

class SmoothPool(MessagePassing):
    def __init__(self, in_features_nodes: int, cluster_num: int, aggr: str="mean"):
        super(SmoothPool,self).__init__(aggr=aggr, flow="source_to_target")
        self.project_vectors = torch.nn.Sequential(
                        Linear(2*in_features_nodes, cluster_num, bias=False),
                        Sparsemax(),
                        )
        self.bn = BatchNorm1d(in_features_nodes)

    def message(self, x_j: Tensor, x_i: Tensor) -> Tensor:
        msg = torch.abs(x_i - x_j)
        return msg
    
    def update(self, inputs: Tensor) -> Tensor:
        return inputs
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):
        
        smoothness = self.propagate(edge_index, x=x)
        x_with_smoothness = torch.cat((x, smoothness), dim=1)
        
        cluster_assignment = self.project_vectors(x_with_smoothness)
        
        batch_num = batch.max().item() + 1
        dense_x, node_mask = to_dense_batch(x, batch)
        cluster_assignment, node_mask = to_dense_batch(cluster_assignment, batch)
        cluster_assignment_t = cluster_assignment.transpose(-2,-1)

        

        dense_adj_with_attr = to_dense_adj(edge_index, batch)
        
        new_dense_adj = cluster_assignment_t@dense_adj_with_attr@cluster_assignment
        new_dense_x = cluster_assignment_t@dense_x

        edge_index, _ = dense_to_sparse(new_dense_adj)

        num_nodes = new_dense_x.size()[1]

        batch = [i for i in range(batch_num) for j in range(num_nodes)]
        batch = torch.tensor(batch, device=x.device)

        x = new_dense_x.view(batch_num*num_nodes, -1)
        x = self.bn(x)

        return x, edge_index, batch


"""
smoothpool with edge attribute, which is not completed 
class SmoothEPool(SmoothPool):
    def __init__(self, in_features_nodes: int, in_features_edge: int, cluster_num: int, aggr: str="add"):
        super().__init__(in_features_nodes, cluster_num, aggr=aggr)

        self.nn_smoothness_gate = torch.nn.Sequential(
                        Linear(in_features_edge, 1),
                        Sigmoid()
        )
        
    
    def message(self, x_j: Tensor, x_i: Tensor, smoothness_gate: Tensor, attention_score: Tensor) -> Tensor:
        msg = torch.abs(x_i - x_j)*(smoothness_gate.unsqueeze(-1)*attention_score.unsqueeze(-1))
        
        return msg

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor, edge_attr:Tensor):

        smoothness_gate = self.nn_smoothness_gate(edge_attr).squeeze(-1)
        attention_score = smoothness_attention(x, edge_index, smoothness_gate)
        smoothness = self.propagate(edge_index, x=x, smoothness_gate=smoothness_gate, attention_score=attention_score) 
        x_with_smoothness = torch.cat((x, smoothness), dim=1)
        
        # calculate assignemnt matrix
        cluster_assignment = self.project_vectors(x_with_smoothness)

        # convert to dense format
        batch_num = batch.max().item() + 1
        dense_x, node_mask = to_dense_batch(x, batch)
        cluster_assignment, node_mask = to_dense_batch(cluster_assignment, batch)
        cluster_assignment_t = cluster_assignment.transpose(-2,-1)

        # encode edge attributes to low dimension space
        # edge_attr = self.nn_encode_edge_attr(edge_attr)
        '''
        dense_adj_with_attr = to_dense_adj(edge_index, batch, edge_attr=edge_attr)
        dense_adj_with_attr = dense_adj_with_attr.permute(3, 0, 1, 2)
        dense_adj_with_attr = (cluster_assignment_t@dense_adj_with_attr@cluster_assignment)
        dense_x = cluster_assignment_t@dense_x

        dense_adj_with_attr = dense_adj_with_attr.permute(1, 2, 3, 0)
      
        edge_index, edge_attr = dense_to_sparse_with_attr(dense_adj_with_attr)
        

        # decode edge attributes
        edge_attr = self.nn_decode_edge_attr(edge_attr)
        '''
        dense_adj_with_attr = to_dense_adj(edge_index, batch)
        
        new_dense_adj = cluster_assignment_t@dense_adj_with_attr@cluster_assignment
        new_dense_x = cluster_assignment_t@dense_x

        edge_index, _ = dense_to_sparse(new_dense_adj)


        num_nodes = new_dense_x.size()[1]

        batch = [ i for i in range(batch_num) for j in range(num_nodes)]
        batch = torch.tensor(batch, device=x.device)

        x = new_dense_x.view(batch_num*num_nodes, -1)

        x = self.bn(x)

        return x, edge_index, batch
"""