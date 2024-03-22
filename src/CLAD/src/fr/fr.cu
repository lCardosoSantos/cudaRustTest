// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#include <cstdint>

#include "fr.cuh"
#include "mul256.cuh"
#include "fr_redc.cuh"
#include "fr_reduce4.cuh"

__host__ __device__ fr_t::fr_t(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
{
    _[0] = x0;
    _[1] = x1;
    _[2] = x2;
    _[3] = x3;
}

__host__ __device__ void fr_set(fr_t &z, uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
{
    z = fr_t(x0, x1, x2, x3);
}

__device__ void fr_t::to_mont(uint64_t &z0, uint64_t &z1, uint64_t &z2, uint64_t &z3)
{
    uint64_t z4, z5, z6, z7;

    // Multiply by R^2 mod r = 2^512 mod r

    mul256(
        z0, z1, z2, z3, z4, z5, z6, z7,
        z0, z1, z2, z3,
        0x1BB8E645AE216DA7,
        0x53FE3AB1E35C59E3,
        0x8C49833D53BB8085,
        0x0216D0B17F4E44A5
    );

    fr_redc(z0, z1, z2, z3, z4, z5, z6, z7);
}

__device__ void fr_t::from_mont(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
{
    uint64_t z0, z1, z2, z3, z4, z5, z6, z7;

    z0 = x0;
    z1 = x1;
    z2 = x2;
    z3 = x3;
    z4 = 0;
    z5 = 0;
    z6 = 0;
    z7 = 0;

    fr_redc(z0, z1, z2, z3, z4, z5, z6, z7);
    fr_reduce4(z0, z1, z2, z3, z0, z1, z2, z3);

    _[0] = z0;
    _[1] = z1;
    _[2] = z2;
    _[3] = z3;
}

__device__ void fr_t::print() const
{
    uint64_t x0, x1, x2, x3;
    fr_reduce4(x0, x1, x2, x3, _[0], _[1], _[2], _[3]);
    printf("#x%016lx%016lx%016lx%016lx", x3, x2, x1, x0);
}

// vim: ts=4 et sw=4 si
