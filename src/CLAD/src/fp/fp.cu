// Copyright 2024 Dag Arne Osvik
// Copyright 2024 Luan Cardoso dos Santos

#include "fp.cuh"

__host__ __device__ fp_t::fp_t(uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
{
    fp_set(*this, x0, x1, x2, x3);
}

__host__  __device__ void fp_set(fp_t &z, uint64_t x0, uint64_t x1, uint64_t x2, uint64_t x3)
{
    // TODO: convert to Montgomery form

    z[0] = x0;
    z[1] = x1;
    z[2] = x2;
    z[3] = x3;
}

__host__  __device__ void fp_t::to_uint64_t(uint64_t &x0, uint64_t &x1, uint64_t &x2, uint64_t &x3)
{
    uint64_t t[4];
    t[0] = _[0];
    t[1] = _[1];
    t[2] = _[2];
    t[3] = _[3];

    // TODO: convert from Montgomery form

    x0 = t[0];
    x1 = t[1];
    x2 = t[2];
    x3 = t[3];
}

// vim: ts=4 et sw=4 si

