// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include "fr.cuh"
#include "fr_reduce4.cuh"

__device__ __noinline__ void fr_reduce4(fr_t &z, const fr_t &x)
{
    fr_reduce4(
        z[0], z[1], z[2], z[3],
        x[0], x[1], x[2], x[3]
    );
}

// vim: ts=4 et sw=4 si
