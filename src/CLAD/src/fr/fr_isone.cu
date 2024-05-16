// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include "fr.cuh" 

__device__ bool fr_is1(const fr_t &x){
    // #warning Temporary implementation for testing
    if(x[0] == 1 && (x[1] | x[2] | x[3]) == 0) return true;
    return false;
}
