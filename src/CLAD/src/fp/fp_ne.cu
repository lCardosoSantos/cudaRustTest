#include "fp.cuh" 

__device__ bool fp_ne(const fp_t &x, const fp_t &y){
    return !fp_eq(x, y);

}
