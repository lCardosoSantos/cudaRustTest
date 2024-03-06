#include "testFields.cuh"
/**
 * @brief Test for squaring on Fp. Checks for self consistency:
 * 
 * (x+n)^2 == x^2 + 2nx + n^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
__global__ bool TestFieldSqr(bool result, T *testval, const size_t testsize){
    #warning Function not implemented: TestFieldSqr
}

/**
 * @brief Test for squaring on Fp. Checks for self consistency:
 * 
 * (x+y)^2 == x^2 + 2xy + y^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
__global__ bool TestFieldSqr2(bool result, T *testval, const size_t testsize){
    #warning Function not implemented: TestFieldSqr2
}
