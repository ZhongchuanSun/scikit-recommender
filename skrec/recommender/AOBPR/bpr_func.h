#ifndef BPR_FUNC_H
#define BPR_FUNC_H
#include <cmath>

float sigmoid(float x)
{
    return 1.0 / (1 + exp(-x));
}

float inner_product(float* a_ptr, float* b_ptr, int n_dim)
{
    float result = 0.0;
    for(int dim=0; dim<n_dim; dim++)
        result += a_ptr[dim] * b_ptr[dim];
    return result;
}

void bpr_update_one_step(float* u_ptr, float* i_ptr, float* j_ptr, float* ib_ptr, float* jb_ptr,
                         int n_dim, float lr, float reg, float weight=1.0)
{
    float xui = inner_product(u_ptr, i_ptr, n_dim) + ib_ptr[0];
    float xuj = inner_product(u_ptr, j_ptr, n_dim) + jb_ptr[0];

    float xuij = xui - xuj;
//    float vals = -log(sigmoid(xuij));
    float cmg = sigmoid(-xuij) * weight;

    for(int d=0; d<n_dim; d++)
    {
        u_ptr[d] += lr * (cmg*(i_ptr[d]-j_ptr[d]) - reg*u_ptr[d]);
        i_ptr[d] += lr * (cmg*u_ptr[d] - reg*i_ptr[d]);
        j_ptr[d] += lr * (cmg*(-u_ptr[d]) - reg*j_ptr[d]);
    }

    ib_ptr[0] += lr * (cmg - reg*ib_ptr[0]);
    jb_ptr[0] += lr * (-cmg - reg*jb_ptr[0]);
}

#endif