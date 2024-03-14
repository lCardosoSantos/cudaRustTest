#include "testG1.cuh"

/**
 * @brief Test for multiplication vs adition
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestMul(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t x, a, b;
    fr_t fr; 

    static int TESTMUL_INNER_LOOP = 1024;
    
    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(x, testval[i]);
        g1p_cpy(a, x);
        for(int j=0; j < TESTMUL_INNER_LOOP; j++){

            g1p_add(a, a, x);
            fr[0] = j; 
            g1p_mul(b, x, fr); 

            if(g1p_neq(a, b)){
                pass = false;
                if (verbosity >= PRINT_MESSAGES){
                    printf("%d.%d: FAILED\n", i, j);
                    printf("sum == mult \n");
                    g1p_print("x:   ", x);
                    g1p_print("sum:   ", a);
                    g1p_print("mul:   ", b);
                }
            ++count;
            if (errorOnce) break;
            }
            if(errorOnce && !pass) break;
        }
    }

    TEST_EPILOGUE;
}
