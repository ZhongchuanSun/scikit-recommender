__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["inner_product", "euclidean_distance", "l2_distance",
           "bpr_loss", "l2_loss"]

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
import torch.nn.functional as F


def inner_product(a: Tensor, b: Tensor, dim: int=-1) -> Tensor:
    return torch.sum(a*b, dim=dim)


def euclidean_distance(a: Tensor, b: Tensor, dim: int=-1) -> Tensor:
    return torch.norm(a-b, p=None, dim=dim)


def l2_distance(a: Tensor, b: Tensor, dim: int=-1) -> Tensor:
    return euclidean_distance(a, b, dim)


def sp_mat_to_sp_tensor(sp_mat: sp.spmatrix) -> Tensor:
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


def bpr_loss(y_pos: Tensor, y_neg: Tensor):
    return -F.logsigmoid(y_pos - y_neg)


def l2_loss(*weights):
    """L2 loss
    Compute  the L2 norm of tensors without the `sqrt`:
        output = sum([sum(w ** 2) / 2 for w in weights])
    Args:
        *weights: Variable length weight list.

    """
    return 0.5 * sum([torch.sum(torch.pow(w, 2)) for w in weights])
