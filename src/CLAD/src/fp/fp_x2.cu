// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#include <cassert>
#include <stdio.h>

#include "fp.cuh"
#include "add320.cuh"
#include "fp_reduce5.cuh"

__noinline__ __device__ void fp_x2(fp_t &z, const fp_t &x) {
    uint64_t
        z0, z1, z2, z3, z4,
        x0 = x[0],
        x1 = x[1],
        x2 = x[2],
        x3 = x[3];

    add320(z0, z1, z2, z3, z4,  x0, x1, x2, x3,  0,  x0, x1, x2, x3,  0);

    fp_reduce5(z0, z1, z2, z3, z0, z1, z2, z3, z4);

    z[0] = z0;
    z[1] = z1;
    z[2] = z2;
    z[3] = z3;

    // if(z4 != 0){
    //     field_print("x", x);
    //     field_print("z", z);
    //     printf("z4 %x\n", z4);
    // }
    
}

// vim: ts=4 et sw=4 si
