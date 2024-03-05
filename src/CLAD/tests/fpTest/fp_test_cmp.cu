#include "fp_test.cuh"

/**
 * @brief Test for the comparison function in Fp; checks for inconsistencies in the 
 * following properties:
 * 
 * eq(x,x) != neq(x,x)
 * neq(x,x) == false
 * neq(x,y) == true
 * eq(x,x) == true
 * eq(x,y) == false
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
extern "C"__global__ bool FpTestCmp(bool result, testval_t *testval, const size_t testsize){
    #warning Function not implemented: FpTestCmp
}
