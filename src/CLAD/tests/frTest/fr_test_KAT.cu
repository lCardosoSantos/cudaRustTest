#include "fr_test.cuh"

/**
 * @brief Test for fr functions using KAT
 * 
 * Tests: fr_copy, fr_reduce6, fr_eq, fr_neq, fr_neg, fr_x2, fr_x3, fr_add, fr_sub,
 * fr_sqr, fr_mul, fr_inv.
 * 
 * This function also uses a mandelbrot iteration to test squaring and addition.
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestKAT(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestKAT
}

/**
 * @brief Test addition and subtraction in Fr using a fibonacci sequence (chain
 * dependency) from 1 to testsize and back
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FrTestFibonacci(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FrTestKAT
}
