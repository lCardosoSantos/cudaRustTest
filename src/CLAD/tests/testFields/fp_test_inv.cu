#include "testFields.cuh"

/**
 * @brief Test for multiplicative inverse mod p in Fp. 
 *
 * Test for self consistency as x == x*inv(x)*x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
__global__ bool TestFieldInv(bool result, T *testval, const size_t testsize){
    #warning Function not implemented: TestFieldInv
}
