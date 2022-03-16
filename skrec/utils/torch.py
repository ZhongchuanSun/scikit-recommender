__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

__all__ = ["inner_product", "euclidean_distance",
           "l2_distance", "bpr_loss", "l2_loss",
           "sigmoid_cross_entropy",
           "sp_mat_to_sp_tensor", "get_initializer"]

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from functools import partial
from collections import OrderedDict


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


def bpr_loss(y_pos: Tensor, y_neg: Tensor) -> Tensor:
    return -F.logsigmoid(y_pos - y_neg)


def l2_loss(*weights):
    """L2 loss
    Compute  the L2 norm of tensors without the `sqrt`:
        output = sum([sum(w ** 2) / 2 for w in weights])
    Args:
        *weights: Variable length weight list.

    """
    return 0.5 * sum([torch.sum(torch.pow(w, 2)) for w in weights])


def truncated_normal_(tensor, mean=0.0, std=1.0):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/16
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_(mean=0, std=1)
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


class InitArg(object):
    MEAN = 0.0
    STDDEV = 0.01
    MIN_VAL = -0.05
    MAX_VAL = 0.05


_initializers = OrderedDict()
_initializers["normal"] = partial(nn.init.normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["truncated_normal"] = partial(truncated_normal_, mean=InitArg.MEAN, std=InitArg.STDDEV)
_initializers["uniform"] = partial(nn.init.uniform_, a=InitArg.MIN_VAL, b=InitArg.MAX_VAL)
_initializers["he_normal"] = nn.init.kaiming_normal_
_initializers["he_uniform"] = nn.init.kaiming_uniform_
_initializers["xavier_normal"] = nn.init.xavier_normal_
_initializers["xavier_uniform"] = nn.init.xavier_uniform_
_initializers["zeros"] = nn.init.zeros_
_initializers["ones"] = nn.init.ones_


def get_initializer(init_method: str):
    if init_method not in _initializers:
        init_list = ', '.join(_initializers.keys())
        raise ValueError(f"'init_method' is invalid, and must be one of '{init_list}'")
    return _initializers[init_method]


def sigmoid_cross_entropy(y_pre, y_true):
    return F.binary_cross_entropy_with_logits(input=y_pre, target=y_true, reduction="none")
