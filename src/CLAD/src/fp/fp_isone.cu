#include "fp.cuh" 

__device__ __host__ bool is1(const fp_t &x){
    #warning Temporary implementation for testing
    if(x[0] == 1 && (x[1] | x[2] | x[3]) == 0) return true;
    return false;
}
