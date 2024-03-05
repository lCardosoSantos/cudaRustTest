#include "fr_test.cuh"

/**
 * @brief Test for multiply-multiply-add. Compare with standalone
 * implementation of multiplication and addition functions.
 * 
 * mma(v, w, x, y) = add(mul(v, w), mul(x, y))
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestMMA(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestMMA
}
