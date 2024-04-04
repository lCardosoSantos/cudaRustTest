#include "testFields.cuh"
#include "fp.cuh"
#include "fr.cuh"

using namespace fp;
using namespace fr;

/**
 * @brief Test for multiplicative inverse mod p in Fp. 
 *
 * Test for self consistency as x == x*inv(x)*x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldInv(bool &result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    //var declare
    T x, y, z, a;

    for (int i=0; pass && i<testsize; i++){
        cpy(x, testval[i]);
        inv(y, x); //y = x^-1
        mul(z, y, x); //y = x^-1 * x 
        mul(a, z, x); //y = x^-1 * x *x


        if(ne(x,a)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                printf("x == x*inv(x)*x");
                field_print("x            : ",  x);
                field_print("x^-1         : ",  y);    
                field_print("x^-1 * x     : ",  z);
                field_print("x^-1 * x * x : ",  a);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}
