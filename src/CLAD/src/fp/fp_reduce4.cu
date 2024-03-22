// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include "fp.cuh"
#include "fp_reduce4.cuh"

__device__ __noinline__ void fp_reduce4(fp_t &z, const fp_t &x)
{
    fp_reduce4(
        z[0], z[1], z[2], z[3],
        x[0], x[1], x[2], x[3]
    );
}

// vim: ts=4 et sw=4 si
