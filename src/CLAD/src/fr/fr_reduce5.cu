// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include "fr.cuh"
#include "fr_reduce5.cuh"

__device__ __noinline__ void fr_reduce5(
    fr_t &z,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4
    )
{
    fr_reduce5(
        z[0], z[1], z[2], z[3],
        x0, x1, x2, x3, x4
    );
}

// vim: ts=4 et sw=4 si
