#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 17:03:22 2021

@author: huwanyang

"""
import numpy as np
import torch

from torch.nn import Sequential, Linear, BatchNorm1d, ReLU
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GINConv
from scipy import sparse


from utils.sparsegen import Sparsegen
from utils.lapool_utils  import cosine_attn, dot_attn, to_tensor, batch_index_select, compute_deg_matrix, inverse_diag_mat


EPS = 1e-8

### adapt La-Pool structure                
class LaPool(torch.nn.Module):
    def __init__(self, in_features, cluster_num, attn='cos', hop=-1, reg_mode=0,
                 concat=False, lap_hop=0, sigma=0.8, **kwargs):
        super(LaPool, self).__init__()
        self.in_features = in_features
        self.cluster_num = cluster_num
        self.net =  GINConv(
                        nn=Sequential(
                            Linear(in_features, in_features),
                            BatchNorm1d(in_features),
                            ReLU()),
                        train_eps=True
                    )

        self.cur_S = None
        self.attn_softmax = Sparsegen(dim=-1, sigma=sigma)
        self.attn_net = cosine_attn  # using cosine attention
        if attn == 'dot':
            self.attn_net = dot_attn

        self.concat = concat
        self.feat_update = GINConv(
                        nn=Sequential(
                            Linear(in_features, in_features),
                            BatchNorm1d(in_features),
                            ReLU()),
                        train_eps=True
                    )
        self.k = cluster_num
        self.reg_mode = reg_mode
        self.hop = hop
        self.lap_hop = lap_hop
        
    def forward(self, x, edge_index, ptr, batch, return_loss=False, **kwargs):
        h = self.net(x,edge_index)
        device = x.device
        #transporm x, edge_index,mask to 3-dimations
        #original lapool compute pooling the leader from each single graph
        batch_size =len(ptr) -1
        max_nodes = int(max(ptr[1:] - ptr[:-1]))
        #split edge list to batch_size*adj_matrixs
        edge_Mtr = to_dense_adj(edge_index, batch) 
        edge_Mtr = torch.eye(max_nodes, device=device) + edge_Mtr
        #transporm h,x to batch_size*mas_nodes*in_features
        h_batch = torch.zeros((batch_size, max_nodes,h.shape[-1]), device=device) 
        x_batch = torch.zeros((batch_size, max_nodes,x.shape[-1]), device=device)
        #create mask
        mask = torch.zeros((batch_size, max_nodes,1), device=device)
        #update mask,h,x
        for p in range(batch_size): 
            num = ptr[p+1] -ptr[p]
            mask[p,:num,0] = 1 
            h_batch[p,:num,:] = h[ptr[p]:ptr[p+1],:]
            x_batch[p,:num,:] = x[ptr[p]:ptr[p+1],:]
            
        mask = mask.type(torch.int)

        S, precluster = self.compute_clusters(adj=edge_Mtr, x=x_batch, h=h_batch, nodes=mask)
        new_edge_inx,new_prt = self.compute_adj(mapper=S, adj=edge_Mtr)
        new_feat,new_batch = self.compute_feats(mapper=S, x=h_batch, precluster=precluster,edge_index = new_edge_inx, ptr = new_prt )

        self.cur_S = S

        new_batch = new_batch.to(device=x.device)
        return new_feat,new_edge_inx,new_batch
    
    def compute_attention(self, adj, nodes, clusters, mask):
        
        attn = self.attn_net(nodes, clusters)

        if self.hop >= 0:  # limit to number of hop
            G = get_path_length(adj, self.hop, strict=False).sum(dim=0)  # number of path
            
        else:  # compute full distance
            with np.errstate(divide='ignore'):
                gpath = np.array(
                    [1 / sparse.csgraph.shortest_path(x, directed=False) for x in adj.clone().detach().cpu().numpy()])
            gpath[np.isinf(gpath)] = 0  # normalized distance (higher, better)
            G = to_tensor(gpath, gpu=False, dtype=torch.float).to(adj.device)
        
        # gives the distance or number of path between each node and the centroid
        last_dim = adj.dim() - 1
        G = batch_index_select(G, last_dim, self.leader_idx) + EPS
        # entry of G should always be zero for non-connected components, so attn will be null for them

        attn = self.attn_softmax(attn * G)
        return attn

    def compute_laplacian(self, adj):
        adj_size = adj.shape[-2]
        deg_mat, adj = compute_deg_matrix(adj)
        if self.reg_mode == 0:
            laplacian = deg_mat - adj  #
        else:
            laplacian = torch.eye(adj_size).to(adj.device) - torch.matmul(inverse_diag_mat(deg_mat), adj)
        
        if self.lap_hop > 1:
            laplacian = torch.matrix_power(laplacian, self.lap_hop)
        return laplacian

    def _select_leader(self, adj, x, nodes=None, **kwargs):

        # Compute the graph Laplacian, then the norm of the Laplacian
        laplacian = self.compute_laplacian(adj).matmul(x)
        laplacian_norm = torch.norm(laplacian, dim=-1)  # b * n

        adj_no_diag = adj - torch.diag_embed(torch.diagonal(adj.permute(1, 2, 0)))
        node_deg, _ = compute_deg_matrix(adj_no_diag, selfloop=False)
        # we want node where the following:
        # \sum(wj xj) / \sum(wj) < xi ==>  D(^-1)AX < X, where A does not have diagonal entry
        nei_laplacian_diff = (
                    laplacian_norm.unsqueeze(-1) - torch.bmm(torch.matmul(inverse_diag_mat(node_deg), adj_no_diag),
                                                             laplacian_norm.unsqueeze(-1))).squeeze(-1)

        # normalize to all strictly positive values
        min_val = torch.min(nei_laplacian_diff, -1, keepdim=True)[0]
        nei_laplacian_normalized = (nei_laplacian_diff - min_val) + torch.abs(min_val)
        

        if nodes is None:
            nodes = torch.ones_like(nei_laplacian_normalized)

        nei_laplacian_normalized = nei_laplacian_normalized * nodes.squeeze(-1).float()  # set unwanted (fake nodes) to 0

        k = self.k
        mask = nei_laplacian_normalized > 0
        if k is None:
            mask = nei_laplacian_diff * nodes.float().squeeze(-1) > 0
            # find best max k for this batch
            k = torch.max(torch.sum(mask, dim=-1))  # maximum number of valid centroid in the batch
            # note that in this we relax the assumption by computing 
            # \sum\limits_{j \neq i} s_i - a_{ij} s_j \big) > 0, and not
            # \forall\; v_j,  s_i - A_{ij} s_j  > 0  as seen in the paper
        _, leader_idx = torch.topk(nei_laplacian_normalized, k=k, dim=-1, largest=True)  # select k best
        self.leader_idx = leader_idx.to(adj.device)
        return leader_idx, mask

    def compute_clusters(self, *, adj, x, h, nodes, **kwargs):
        leader_idx, mask = self._select_leader(adj, h, nodes)
        clusters = batch_index_select(h, 1, leader_idx)
        
        attn = self.compute_attention(adj, h, clusters, mask)
        return attn, clusters

    def _loss(self, adj, mapper):
        
        return 0  # (LLP + LE)

    def compute_adj(self, *, mapper, adj, **kwargs):
        
        adj = mapper.transpose(-2, -1).matmul(adj).matmul(mapper)
        # Remove diagonal
        adj = (1 - torch.eye(adj.shape[-1]).unsqueeze(0)).to(adj.device) * adj
        # cover adj_matrixs to edge list,if adj[a,b] is not equal to 0,then a and b are conneted in new graph
        edge_inx = None 
        max_node = 0
        new_prt  = [max_node]

        device = adj.device
        adj = adj.cpu()

        for i in range(adj.shape[0]):
            if edge_inx == None:
                edge_inx = torch.LongTensor(np.where(adj[i]!=0))
                max_node = torch.max(edge_inx)+1
                new_prt.append(max_node)
                
            else:
                edge = torch.LongTensor(np.where(adj[i]!=0)) + max_node
                if len(edge[0]) > 1:
                    max_node = torch.max(edge)+1
                edge_inx  = torch.cat((edge_inx,edge),dim = 1)
                new_prt.append(max_node)

           
        edge_inx = edge_inx.to(device=device)
        new_prt = torch.LongTensor(new_prt).to(device=device)
        return edge_inx.type(torch.int64), new_prt

    def compute_feats(self, *, mapper, x, precluster, edge_index, ptr, **kwargs):
        if not self.concat:
            clusters = torch.bmm(mapper.transpose(-2, -1), x)
        else:
            # case where the batchsize is one
            x = x.unsqueeze(0) if x.dim() == 2 else x
            mapper = mapper.unsqueeze(0) if mapper.dim() == 2 else mapper
            precluster = precluster.unsqueeze(0) if precluster.dim() == 2 else precluster
            clusters = torch.cat((torch.bmm(mapper.transpose(-2, -1), x), precluster), dim=-1)
            
        #squeeze features 3d features to 2d, update batch information
        new_feat = clusters[0,:ptr[1],:]
        new_batch = torch.zeros(ptr[1])
        for p in range(1,len(ptr)-1):
            feat = clusters[p,:ptr[p+1]-ptr[p],:]
            new_feat = torch.cat((new_feat,feat))
            batch = torch.zeros(ptr[p+1]-ptr[p])+p
            new_batch = torch.cat((new_batch,batch))
            
        new_feat = self.feat_update(new_feat,edge_index)
        return new_feat,new_batch.type(torch.int64)

    
def get_path_length(adj_mat, k, strict=True):
    prev_mat = adj_mat
    matlist = adj_mat.unsqueeze(0)
    for i in range(2, k + 1):
        prev_mat = torch.bmm(prev_mat, adj_mat)
        if strict:
            no_path = (matlist.sum(dim=0) != 0).byte()
            new_mat = prev_mat.masked_fill(no_path, 0)
        else:
            new_mat = prev_mat
        # there os no point in receiving msg from the same node.
        # new_mat.clamp_max_(1)
        matlist = torch.cat((matlist, new_mat.unsqueeze(0)), dim=0).clamp_max_(1)
    return matlist