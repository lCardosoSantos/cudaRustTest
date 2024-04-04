#include "fp.cuh" 

__device__ bool fp_eq(const fp_t &x, const fp_t &y){
    fp_t t; 
    fp_sub(t, x, y);
    fp_reduce4(t[0], t[1], t[2], t[3],
               t[0], t[1], t[2], t[3]);
    return fp_is0(t);
}
