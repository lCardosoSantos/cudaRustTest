// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include "fp.cuh" 

__device__  bool fp_is1(const fp_t &x){
    // #warning only for debugging
    if(x[0] == 1 && (x[1] | x[2] | x[3]) == 0) return true;
    return false;
}
