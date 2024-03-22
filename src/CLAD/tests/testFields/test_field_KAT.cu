#include "testFields.cuh"

/**
 * @brief Test for fp functions using KAT
 * 
 * Tests: fp_copy, fp_reduce6, fp_eq, fp_neq, fp_neg, fp_x2, fp_x3, fp_add, fp_sub,
 * fp_sqr, fp_mul, fp_inv.
 * 
 * This function also uses a mandelbrot iteration to test squaring and addition.
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldKAT(bool &result, T *testval, const size_t testsize){
    #warning Function not implemented: TestFieldKAT
}

// /**
//  * @brief Test addition and subtraction in Fp using a fibonacci sequence (chain
//  * dependency) from 1 to testsize and back
//  * 
//  * @param testval 
//  * @param testsize 
//  * 
//  * @return bool 
//  */
// template<typename T>
//  __global__ void TestFieldFibonacci(bool &result, T *testval, const size_t testsize){
//     #warning Function not implemented: TestFieldFibonacci
// }
