#include "fr.cuh" 

__device__ bool fr_ne(const fr_t &x, const fr_t &y){
    return !fr_eq(x, y);
}
