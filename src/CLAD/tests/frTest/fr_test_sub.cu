#include "fr_test.cuh"

/**
 * @brief Test for subtraction in Fr.
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @return __global__ 
 */
extern "C"__global__ bool FrTestSub(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestSub
}

/**
 * @brief Check the distributive property of multiplication in Fr (left of subtraction):
 * 
 * a(b-c) = ab-ac
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestSubDistributiveLeft(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestSubDistributiveLeft
}

/**
 * @brief Check the distributive property of multiplication in Fr (right of subtraction):
 * 
 * (a-b)c = ac-bc
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestSubDistributiveRight(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestSubDistributiveRight
}
