#include "fp_test.cuh"

/**
 * @brief Test for subtraction in Fp.
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @return __global__ 
 */
extern "C"__global__ bool FpTestSub(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestSub
}

/**
 * @brief Check the distributive property of multiplication in Fp (left of subtraction):
 * 
 * a(b-c) = ab-ac
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestSubDistributiveLeft(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestSubDistributiveLeft
}

/**
 * @brief Check the distributive property of multiplication in Fp (right of subtraction):
 * 
 * (a-b)c = ac-bc
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestSubDistributiveRight(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestSubDistributiveRight
}
