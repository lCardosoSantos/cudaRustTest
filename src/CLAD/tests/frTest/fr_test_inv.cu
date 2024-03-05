#include "fr_test.cuh"

/**
 * @brief Test for multiplicative inverse mod p in Fr. 
 *
 * Test for self consistency as x == x*inv(x)*x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestInv(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestInv
}
