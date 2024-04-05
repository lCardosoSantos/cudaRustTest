// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#include "fr.cuh"
#include "add320.cuh"
#include "fr_reduce5.cuh"

__noinline__ __device__ void fr_x4(fr_t &z, const fr_t &x) {
    uint64_t
        z0, z1, z2, z3, z4,
        x0 = x[0],
        x1 = x[1],
        x2 = x[2],
        x3 = x[3];

    add320(z0, z1, z2, z3, z4,  x0, x1, x2, x3,  0,  x0, x1, x2, x3,  0);
    add320(z0, z1, z2, z3, z4,  z0, z1, z2, z3, z4,  z0, z1, z2, z3, z4);

    fr_reduce5(z0, z1, z2, z3, z0, z1, z2, z3, z4);

    z[0] = z0;
    z[1] = z1;
    z[2] = z2;
    z[3] = z3;
}

// vim: ts=4 et sw=4 si
