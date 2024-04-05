#include "fr.cuh" 

__device__ __host__ void cpy(fr_t &z, const fr_t &x){
     z = fr_t(x[0], x[1], x[2], x[3]);
}
