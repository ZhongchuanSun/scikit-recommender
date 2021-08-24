# distutils: language = c++
# cython: language_level = 3
__author__ = "Zhongchuan Sun"
__email__ = "zhongchuansun@foxmail.com"

import numpy as np

__all__ = ["is_base_ndarray", "to_base_ndarray"]


def is_base_ndarray(array, dtype=None):
    if not isinstance(array, np.ndarray):
        return False
    elif array.base is not None:
        return False
    elif dtype is None:
        return True
    elif array.dtype == dtype:
        return True
    else:
        return False


def to_base_ndarray(array, dtype=None, copy=False):
    if not is_base_ndarray(array, dtype):
        return np.array(array, dtype=dtype, copy=True)
    else:
        return np.array(array, dtype=dtype, copy=copy)
