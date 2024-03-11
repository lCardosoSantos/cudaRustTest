#include "testFields.cuh"

/**
 * @brief Test for subtraction in Fp.
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @return __global__ 
 */
template<typename T>
 __global__ void TestFieldSub(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldSub
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
template<typename T>
 __global__ void TestFieldSubDistributiveLeft(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldSubDistributiveLeft
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
template<typename T>
 __global__ void TestFieldSubDistributiveRight(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldSubDistributiveRight
}
