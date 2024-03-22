// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#ifndef ADD320_CUH
#define ADD320_CUH

#include <cstdint>

#include "ptx.cuh"

__forceinline__ __device__ void add320(
    uint64_t &z0,
    uint64_t &z1,
    uint64_t &z2,
    uint64_t &z3,
    uint64_t &z4,
    uint64_t x0,
    uint64_t x1,
    uint64_t x2,
    uint64_t x3,
    uint64_t x4,
    uint64_t y0,
    uint64_t y1,
    uint64_t y2,
    uint64_t y3,
    uint64_t y4)
{
    add_cc_u64 (z0, x0, y0);
    addc_cc_u64(z1, x1, y1);
    addc_cc_u64(z2, x2, y2);
    addc_cc_u64(z3, x3, y3);
    addc_u64   (z4, x4, y4);
}

#endif

// vim: ts=4 et sw=4 si
