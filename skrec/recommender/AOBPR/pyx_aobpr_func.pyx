# distutils: language = c++
# cython: cdivision=True
# cython: nonecheck=False
# cython: language_level=3
import numpy as np
cimport numpy as np
from libc.math cimport fabs
from ...utils.py.cython.pyx_utils import to_base_ndarray, is_base_ndarray

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937() # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed) # not worrying about matching the exact int type for seed

    cdef cppclass discrete_distribution[T]:
        discrete_distribution()
        # The following constructor is really a more generic template class
        # but tell Cython it only accepts pointer of float
        discrete_distribution(float* first, float* last)
        T operator()(mt19937 gen)

cdef mt19937 _private_gen = mt19937(2020)


cdef extern from "bpr_func.h":
    float inner_product(float* a_ptr, float* b_ptr, int n_dim)
    float sigmoid(float x)


cdef void bpr_update_one_step(float* u_ptr, float* i_ptr, float* j_ptr,
                              int n_dim, float lr, float reg):
    cdef float xui = inner_product(u_ptr, i_ptr, n_dim)
    cdef float xuj = inner_product(u_ptr, j_ptr, n_dim)
    cdef float xuij = xui - xuj
    cdef float cmg = sigmoid(-xuij)

    cdef int d = 0
    for d in range(n_dim):
        u_ptr[d] += lr * (cmg*(i_ptr[d]-j_ptr[d]) - reg*u_ptr[d])
        i_ptr[d] += lr * (cmg*u_ptr[d] - reg*i_ptr[d])
        j_ptr[d] += lr * (cmg*(-u_ptr[d]) - reg*j_ptr[d])


def aobpr_update(user_arr, pos_item_arr, rank_idx_arr, lr, reg,
                 user_embeds, item_embeds):
    user_arr = to_base_ndarray(user_arr, dtype=np.int32)
    pos_item_arr = to_base_ndarray(pos_item_arr, dtype=np.int32)
    rank_idx_arr = to_base_ndarray(rank_idx_arr, dtype=np.int32)

    if not is_base_ndarray(user_embeds, np.float32) or \
            not is_base_ndarray(item_embeds, np.float32):
        raise ValueError("Parameters must be np.ndarray.")

    _aobpr_update(user_arr, pos_item_arr, rank_idx_arr, lr, reg,
                  user_embeds, item_embeds)


def _sort_factors(item_embeds):
    mean = np.mean(item_embeds, axis=0)
    std = np.std(item_embeds, axis=0)
    sorted_items = np.argsort(-item_embeds, axis=0)
    return sorted_items.astype(np.int32), mean.astype(np.float32), std.astype(np.float32)


cdef int _sample_factor(float *prob_ptr, int n_dim):
    cdef discrete_distribution[int] dd = discrete_distribution[int](prob_ptr, prob_ptr+n_dim)
    return dd(_private_gen)


cdef void _factor_prob(float* u_ptr, float* std_ptr, float* result_ptr, int n_dim):
    # calculate the probability of factors
    cdef:
        float cdf = 0.0
        int f = 0
    for f in range(n_dim):
        result_ptr[f] = fabs(u_ptr[f]) * std_ptr[f]
        cdf += result_ptr[f]
    # normalize
    for f in range(n_dim):
        result_ptr[f] /= cdf


cdef void _aobpr_update(user_arr, pos_item_arr, rank_idx_arr, lr, reg,
                        user_embeds, item_embeds):

    cdef int num_pair = len(user_arr)
    user_ptr = <int *>np.PyArray_DATA(user_arr)
    pos_item_ptr = <int *>np.PyArray_DATA(pos_item_arr)
    rank_idx_ptr = <int *>np.PyArray_DATA(rank_idx_arr)

    u_emb_ptr = <float *>np.PyArray_DATA(user_embeds)
    i_emb_ptr = <float *>np.PyArray_DATA(item_embeds)

    cdef int n_dim = user_embeds.shape[-1]
    cdef int num_items = np.shape(item_embeds)[0]
    cdef int num_loop = num_items*np.log(num_items)

    cdef int user = -1
    cdef int factor = -1
    cdef int pos_item = -1
    cdef int neg_item = -1
    cdef float* u_ptr
    cdef float* i_ptr
    cdef float* j_ptr
    cdef float clr = lr
    cdef float creg = reg
    cdef int rank_idx = -1

    fprob = np.zeros(n_dim, dtype=np.float32)
    cdef float* fprob_ptr = <float *>np.PyArray_DATA(fprob)

    sorted_items, mean, std = _sort_factors(item_embeds)
    sorted_items_ptr = <int *>np.PyArray_DATA(sorted_items)
    mean_ptr = <float *>np.PyArray_DATA(mean)
    std_ptr = <float *>np.PyArray_DATA(std)

    cdef int idx = 0
    for idx in range(num_pair):
        user = user_ptr[idx]
        pos_item = pos_item_ptr[idx]
        rank_idx = rank_idx_ptr[idx]
        if (idx+1) % num_loop == 0:
            sorted_items, mean, std = _sort_factors(item_embeds)
            sorted_items_ptr = <int *>np.PyArray_DATA(sorted_items)
            mean_ptr = <float *>np.PyArray_DATA(mean)
            std_ptr = <float *>np.PyArray_DATA(std)

        u_ptr = u_emb_ptr+user*n_dim
        _factor_prob(u_ptr, std_ptr, fprob_ptr, n_dim)
        # calculate the probability of factors

        # sampling negative item
        factor = _sample_factor(fprob_ptr, n_dim)
        if u_emb_ptr[user*n_dim+factor] > 0:
            neg_item = sorted_items_ptr[rank_idx*n_dim + factor]
        else:
            neg_item = sorted_items_ptr[(num_items-rank_idx-1)*n_dim + factor]

        # u_ptr = u_emb_ptr+user*n_dim
        i_ptr = i_emb_ptr+pos_item*n_dim
        j_ptr = i_emb_ptr+neg_item*n_dim

        bpr_update_one_step(u_ptr, i_ptr, j_ptr, n_dim, clr, creg)
