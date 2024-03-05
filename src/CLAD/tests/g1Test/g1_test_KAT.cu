#include "g1_test.cuh"

/**
 * @brief Test operation over G1 using KAT and self consistency:
 * 
 * inf==inf
 * inf+inf == inf
 * G+0 == 0+G == G
 * G+G == 2*G
 * 2*G == 2*G with KAT
 * G+2*G == 3*G with KAT
 * 2*2*2*3G == 24G with KAT
 * 24G-2G+3G == 25G with KAT
 * 25*G == 25G with KAT
 * addsub(2G, G) == 3G, G with KAT
 * addsub(G, G) = (2G, 2G) (dbl and add)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool G1TestKAT(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: G1TestKAT
}

/**
 * @brief Test addition and multiplication using a fibonacci sequence (cascading
 * data dependency)
 * 
 * @return void 
 */
extern "C"__global__ bool G1TestFibonacci(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: G1TestFibonacci
}
