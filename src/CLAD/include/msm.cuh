#pragma once

#include "g1.cuh"
#include "fr.cuh"

typedef struct g1a_r_t{
    uint64_t x[4];
    uint64_t y[4];
}g1ar_t;

//function interfacing with rust
extern "C" void msm(g1a_t *out, const g1a_t *points, const fr_t *scalars, const size_t nPoints);
