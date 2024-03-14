#pragma once

#include "g1.cuh"
#include "fr.cuh"

//function interfacing with rust
extern "C" void msm(g1a_t *out, const g1a_t *points, const fr_t *scalars, const size_t nPoints);
