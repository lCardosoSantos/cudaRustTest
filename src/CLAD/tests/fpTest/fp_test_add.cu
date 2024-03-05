#include "fp_test.cuh"

/**
 * @brief Test for addition in Fp
 * 
 * 2x + x == 3x
 * 
 * @param testval 
 * @return void
 */
extern "C"__global__ bool FpTestAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestAdd
}

/**
 * @brief Test for the commutative property of addition in Fp
 * 
 * x+y == y+x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestCommutativeAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestCommutativeAdd
}

/**
 * @brief Test for the associative property of addition in Fp
 * 
 * (x+y)+z == x+(y+z)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestAssociativeAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestAssociativeAdd
}

/**
 * @brief Check the distributive property of multiplication in Fp (left of addition):
 * 
 * a(b+c) = ab+ac
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestAddDistributiveLeft(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestAddDistributiveLeft
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
extern "C"__global__ bool FpTestAddDistributiveRight(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestAddDistributiveRight
}
