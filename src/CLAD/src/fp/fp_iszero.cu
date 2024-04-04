#include "fp.cuh" 

__device__  bool fp_is0(const fp_t &x){
    
    if( (x[0] | x[1] | x[2] | x[3]) == 0) 
        return true;

    return false;
    
}
