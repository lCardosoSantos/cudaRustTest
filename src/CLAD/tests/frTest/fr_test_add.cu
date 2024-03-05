#include "fr_test.cuh"

/**
 * @brief Test for addition in Fr
 * 
 * 2x + x == 3x
 * 
 * @param testval 
 * @return void
 */
extern "C"__global__ bool FrTestAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestAdd
}

/**
 * @brief Test for the commutative property of addition in Fr
 * 
 * x+y == y+x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestCommutativeAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestCommutativeAdd
}

/**
 * @brief Test for the associative property of addition in Fr
 * 
 * (x+y)+z == x+(y+z)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestAssociativeAdd(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestAssociativeAdd
}

/**
 * @brief Check the distributive property of multiplication in Fr (left of addition):
 * 
 * a(b+c) = ab+ac
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestAddDistributiveLeft(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestAddDistributiveLeft
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
extern "C"__global__ bool FrTestAddDistributiveRight(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestAddDistributiveRight
}
