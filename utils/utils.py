#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""""
@author: Yizhu Chen
"""

import torch
from sparsemax import Sparsemax

def dense_to_sparse(adj):
    r"""Converts a dense adjacency matrix to a sparse adjacency matrix defined
    by edge indices and edge attributes.
    Args:
        adj (Tensor): The dense adjacency matrix.
     :rtype: (:class:`LongTensor`, :class:`Tensor`)
    
    @author: Matthias Fey
    """
    assert adj.dim() >= 2 and adj.dim() <= 3
    assert adj.size(-1) == adj.size(-2)

    index = adj.nonzero(as_tuple=True)
    #print(index)
    edge_attr = adj[index]

    if len(index) == 3:
        batch = index[0] * adj.size(-1)
        index = (batch + index[1], batch + index[2])

    return torch.stack(index, dim=0), edge_attr

def dense_to_sparse_with_attr(adj):
    r"""
    @author: Matthias Fey
    """
    adj2 = adj.abs().sum(dim=-1).detach()  # Find non-zero edges in `adj` with multi-dimensional edge_features
    index = adj2.nonzero(as_tuple=True)
    edge_attr = adj[index]
    batch = index[0] * adj.size(-2)
    index = (batch + index[1], batch + index[2])
    edge_index = torch.stack(index, dim=0)
    return edge_index, edge_attr

if __name__ == "__main__":
    # adj is a batched dense adjacency matrix of size (2,2,2,3), 
    # which means batch size is 2, there are two nodes in each graph and the dimension of edge attributes is 3 
    adj = torch.tensor([
                [[[1,1,1], [0,1,0]],
                 [[1,0,1], [0,1,0]]],
                [[[1,1,1], [0,1,0]],
                 [[1,0,1], [0,1,0]]],
            ])
    edge_index, edge_attr = dense_to_sparse_with_attr(adj)
    print(edge_index)
    print(edge_index.max())
    feature = torch.tensor([[2.5,2,3],[3,0,6]])
    sparsemax = Sparsemax()
    print(sparsemax(feature))
    