#include "testFields.cuh"

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
template<typename T>
__global__ bool TestFieldCmp(bool result, T *testval, const size_t testsize){
    #warning Function not implemented: TestFieldCmp
}
