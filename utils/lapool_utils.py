"""
Comes form the source code of lapool

"""

import torch
import numpy as np

def cosine_attn(x1, x2, eps=1e-8):
    """Compute attention using cosine similarity"""
    w1 = x1.norm(p=2, dim=-1, keepdim=True)
    w2 = x2.norm(p=2, dim=-1, keepdim=True)
    return torch.matmul(x1, x2.transpose(-2, -1)) / (w1 * w2.transpose(-2, -1)).clamp(min=eps)

def dot_attn(x1, x2, **kwargs):
    attn = torch.matmul(x1, x2.transpose(-2, -1)) # B, N, M 
    return attn / np.sqrt(x1.shape[-1])

def batch_index_select(input, dim, index):
    #print('input',input.shape, index.shape)
    views = [input.shape[0]] + \
        [1 if i != dim else -1 for i in range(1, len(input.shape))]
    #print('views',views)
    expanse = list(input.shape)
    expanse[0] = -1
    expanse[dim] = -1
    index = index.view(views).expand(expanse)
    #print('output',input.shape, dim, index.shape,index)
    #print(torch.gather(input, dim, index).shape)
    return torch.gather(input, dim, index)

def to_tensor(x, gpu=True, dtype=None):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    else:
        x = torch.Tensor(x)
    if dtype is not None:
        x = x.type(dtype)
    if torch.cuda.is_available() and gpu:
        x = x.cuda()
    return x

def compute_deg_matrix(adj_mat, inv=False, selfloop=False):
    r"""
    Compute the inverse deg matrix from a tensor adjacency matrix

    Arguments
    ----------
        adj_mat: `torch.FloatTensor` of size (...)xNxN
            Input adjacency matrices (should not be normalized) that corresponds to a graph
        inv : bool, optional
            Whether the inverse of the computed degree matrix should be returned
            (Default value = False)
        selfloop: bool, optional
            Whether to add selfloop to the adjacency matrix or not

    Returns
    -------
        deg_mat: `torch.Tensor`
            Degree matrix  of the input graph    
    """
    if selfloop:
        adj_mat = torch.eye(adj_mat.shape[-1], device=adj_mat.device).expand_as(
            adj_mat) + adj_mat.clone()
    elif adj_mat.is_sparse:
        adj_mat = adj_mat.to_dense()
    deg_mat = torch.sum(adj_mat, dim=-2)
    if inv:
        # relying on the fact that it is a diag mat
        deg_mat = torch.pow(deg_mat, -0.5)
        deg_mat[torch.isinf(deg_mat)] = 0
    deg_mat = torch.diag_embed(deg_mat)
    return deg_mat, adj_mat


def inverse_diag_mat(mat, eps=1e-8):
    # add jitter to the matrix diag
    jitter = torch.eye(mat.shape[-1], device=mat.device).expand_as(mat)
    if torch.all(mat.masked_select(jitter.bool())>0):
        return mat.inverse() 
    return torch.inverse(jitter*eps + mat)