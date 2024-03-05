#include "fr_test.cuh"
/**
 * @brief Test for squaring on Fr. Checks for self consistency:
 * 
 * (x+n)^2 == x^2 + 2nx + n^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestSqr(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestSqr
}

/**
 * @brief Test for squaring on Fr. Checks for self consistency:
 * 
 * (x+y)^2 == x^2 + 2xy + y^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestSqr2(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestSqr2
}
