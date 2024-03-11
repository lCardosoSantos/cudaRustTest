#include "testFields.cuh"

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
template<typename T>
 __global__ void TestFieldMMA(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldMMA
}
