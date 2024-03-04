#include "g1.cuh"


//dummy function for debug
extern "C" void clad();

//function interfacing with rust
extern "C" void clad_msm(g1a_t *out, const g1a_t *points, const fr_t *scalars, const size_t nPoints);
