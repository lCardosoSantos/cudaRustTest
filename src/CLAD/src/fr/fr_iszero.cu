// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include "fr.cuh" 

__device__ bool fr_is0(const fr_t &x){
    if( (x[0] | x[1] | x[2] | x[3]) == 0) 
        return true;

    return false;
}
