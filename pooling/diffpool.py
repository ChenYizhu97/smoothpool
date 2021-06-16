#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
@author: Yizhu Chen
"""

import torch
from torch import Tensor
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from utils.utils import dense_to_sparse
from torch_geometric.utils import to_dense_adj, to_dense_batch

class Diffpool(torch.nn.Module):
    def __init__(self, in_features_node: int,  num_clusters: int):
        super(Diffpool, self).__init__()

        self.in_features_node = in_features_node

        self.pool = dense_diff_pool
        self.nn = DenseSAGEConv(in_features_node, num_clusters)
    
    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor):

        dense_adj = to_dense_adj(edge_index, batch).squeeze(-1)
        dense_x, mask = to_dense_batch(x, batch)

        batch_num = batch.max().item() + 1

        
        assignment_matrix = self.nn(dense_x, dense_adj)
        new_dense_x, new_dense_adj, link_loss, ent_loss = self.pool(dense_x, dense_adj, assignment_matrix, mask)

        node_num = new_dense_x.size()[1]

        edge_index, _ = dense_to_sparse(new_dense_adj)
        
        x = new_dense_x.view(batch_num*node_num, -1)
       
        batch = [i for i in range(batch_num) for j in range(node_num)]
        batch = torch.tensor(batch, device=x.device, dtype=torch.long)

        return x, edge_index, batch, link_loss, ent_loss