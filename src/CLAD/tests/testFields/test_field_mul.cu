#include "testFields.cuh"

/**
 *  @brief Test self consistency in multiplication by constant:
 * 
 * 2(4x) = =8x
 * 2(2(2(2(2(2x))))) == 4(4(4x)) == 8(8x)
 * 3(4x) == 12(x)
 * 3(3(3(2(4(8x))))) == 12(12(12x))
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldMulConst(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldMulConst
}

/**
 * @brief Multiplication test, using different values for different threads.
 * 
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldMul(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldMul
}

/**
 * @brief Test for the commutative property of addition
 * 
 * x*y == y*x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldCommutativeMul(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldCommutativeMul
}

/**
 * @brief Test for the associative property of multiplication
 * 
 * (x*y)*z == x*(y*z)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldAssociativeMul(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAssociativeMul
}
