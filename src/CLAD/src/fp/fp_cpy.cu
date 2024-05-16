// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include "fp.cuh" 

__device__ __host__ void cpy(fp_t &z, const fp_t &x){
    z = fp_t(x[0], x[1], x[2], x[3]);
}
