#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
@author: Yizhu Chen, Wanyang Hu
"""

from torch.nn import Linear, BatchNorm1d, ReLU, Sigmoid
from torch.nn.modules.container import Sequential
import torch.nn as nn
from torch_geometric.nn import TopKPooling, global_mean_pool, GINConv
from pooling.smoothpool import SmoothPool
from pooling.diffpool import Diffpool
from pooling.lapool import LaPool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch_geometric.data import Batch


class GNN(nn.Module):
    def __init__(self, in_features: int, out_features: int, atom_encoder: AtomEncoder, bond_encoder: BondEncoder, 
                    num_of_conv: int=5, position_of_hpool: int=3, hpool: str="topk", C: int=10):
        super(GNN, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.num_of_conv = num_of_conv
        self.position_of_hpool = position_of_hpool

        self.in_features = in_features
        self.out_features = out_features

        self.atom_encoder = atom_encoder
        self.bond_encoder = bond_encoder

        self.hpool_archi = hpool
        self.C = C
        
        
        for layer in range(self.num_of_conv):
            self.conv_layers.append(
                GINConv(nn=Sequential(
                    Linear(in_features, in_features),
                    BatchNorm1d(in_features),
                    ReLU()),
                train_eps=True
                )
            )

        if self.hpool_archi == "topk":
            self.hpool = TopKPooling(in_features, 0.5)
        
        if self.hpool_archi == "diffpool":
            self.hpool = Diffpool(in_features, self.C)

        if self.hpool_archi == "smoothpool":    
            self.hpool = SmoothPool(in_features, self.C)

        if self.hpool_archi == "lapool": 
            self.hpool = LaPool(in_features, self.C)

        self.gpool = global_mean_pool
        
        self.out_layers = nn.Sequential(
                                        Linear(in_features, out_features),
                                        Sigmoid()
                                        )
    
    def forward(self, batch_data: Batch):
        x, edge_index, edge_attr, ptr, batch = batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.ptr, batch_data.batch

        x = self.atom_encoder(x)
        edge_attr = self.bond_encoder(edge_attr)

        for layer in range(self.num_of_conv):
            if layer == self.position_of_hpool:
                if self.hpool_archi == "topk":
                    x, edge_index, edge_attr, batch, _, _ = self.hpool(x, edge_index, batch=batch, edge_attr=edge_attr)
                if self.hpool_archi == "diffpool":
                    x, edge_index, batch, link_loss, ent_loss = self.hpool(x, edge_index, batch) 
                if self.hpool_archi == "smoothpool":
                    x, edge_index, batch = self.hpool(x, edge_index, batch)

                if self.hpool_archi == "lapool": 
                    x, edge_index, batch = self.hpool(x, edge_index, ptr,batch)

            x = self.conv_layers[layer](x, edge_index)        

        x = self.gpool(x, batch)
        x = self.out_layers(x)

        if self.hpool_archi == "diffpool": 
            return x, link_loss, ent_loss
        else:
            return x
