#include "testG1.cuh"

/**
 * @brief Test for point addition
 * 
 * 2x == 3x - x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestSub(bool *result, testval_t *testval, const size_t testsize){
        
    TEST_PROLOGUE;

    g1p_t x, l, r;
    fr_t three = fr_t(3); 

    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(x, testval[i]);
        g1p_add(l, x, x);

        g1p_mul(r, x, three);
        g1p_sub(r, r, x);

        if(g1p_neq(l, r)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                g1p_print("x:   ", x);
                g1p_print("2x:  ", l);
                g1p_print("3x: ", r);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}
