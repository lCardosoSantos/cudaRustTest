#include "testFields.cuh"

/**
 * @brief Test for addition in Fp
 * 
 * 2x + x == 3x
 * 
 * @param testval 
 * @return void
 */
template<typename T>
 __global__ void TestFieldAdd(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAdd
}

/**
 * @brief Test for the fp_addsub kernel.
 * 
 * Tests using the following properties:
 * 
 * f(x,y) = (x+y,x-y)
 * f(f(x,y)) = (2x, 2y)
 * 
 * @param testval s
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldAddsub(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAddsub
}


/**
 * @brief Test for the commutative property of addition in Fp
 * 
 * x+y == y+x
 * 
 * @param testval s
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldCommutativeAdd(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldCommutativeAdd
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
template<typename T>
 __global__ void TestFieldAssociativeAdd(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAssociativeAdd
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
template<typename T>
 __global__ void TestFieldAddDistributiveLeft(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAddDistributiveLeft
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
 __global__ void TestFieldAddDistributiveRight(bool result, T *testval, const size_t testsize){
    //#warning Function not implemented: TestFieldAddDistributiveRight
}
