// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#include <cstdint>

#include "fp.cuh"
#include "mul256.cuh"
#include "fp_reduce8.cuh"

/**
 * @brief Multiplies two Fp residues x and y, stores the result in z.
 *
 * @param[out] z
 * @param[in] x
 * @param[in] y
 * @return void
 */
__device__ void fp_mul(fp_t &z, const fp_t &x, const fp_t &y) {

    uint64_t
        z0, z1, z2, z3, z4, z5, z6, z7,
        x0, x1, x2, x3,
        y0, y1, y2, y3;

    x0 = x[0];
    x1 = x[1];
    x2 = x[2];
    x3 = x[3];

    y0 = y[0];
    y1 = y[1];
    y2 = y[2];
    y3 = y[3];

    mul256(
        z0, z1, z2, z3, z4, z5, z6, z7,
        x0, x1, x2, x3,
        y0, y1, y2, y3
    );

    fp_reduce8(z0, z1, z2, z3, z0, z1, z2, z3, z4, z5, z6, z7);

    z[0] = z0;
    z[1] = z1;
    z[2] = z2;
    z[3] = z3;
}

// vim: ts=4 et sw=4 si
