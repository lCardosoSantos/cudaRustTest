#include "fr.cuh" 

__device__ bool fr_eq(const fr_t &x, const fr_t &y){
    fr_t t; 
    fr_sub(t, x, y);
    fr_reduce4(t[0], t[1], t[2], t[3],
               t[0], t[1], t[2], t[3]);
    return fr_is0(t);
}
