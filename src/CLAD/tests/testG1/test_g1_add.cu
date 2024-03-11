#include "testG1.cuh"


/**
 * @brief Test for point addition using self consistency
 * 
 * 2x == x + x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestAdd(bool *result, testval_t *testval, const size_t testsize){
    
    TEST_PROLOGUE;

    g1p_t x, l, r;
    fr_t two = fr_t(2); 

    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(x, testval[i]);
        g1p_add(l, x, x);

        g1p_mul(r, x, two);

        if(g1p_neq(l, r)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                g1p_print("x:   ", x, stderr);
                g1p_print("2x:  ", l, stderr);
                g1p_print("x+x: ", r, stderr);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for point addition-subtraction combo function
 * 
 * check if addsub(addsub((p,q))) = (2p, 2q)    
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestAddsub(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t p, q, a, b, p2, q2;

    for (int i=0; pass && i<(testsize/2); i++){
        g1p_cpy(p, testval[2*i]);
        g1p_cpy(q, testval[2*i+1]);

        g1p_cpy(a, p);
        g1p_cpy(b, q);
        g1p_dbl(p2, p);
        g1p_dbl(q2, q);

        g1p_addsub(a, b);
        g1p_addsub(a, b); //a==p2, b==q2

        if(g1p_neq(p2, a) || g1p_neq(q2, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "addsub(addsub((p,q))) -> (a,b); a==2p, b==2q \n");
                g1p_print("p:   ", p, stderr);
                g1p_print("q:   ", q, stderr);
                g1p_print("a:   ", a, stderr);
                g1p_print("b:   ", b, stderr);
                g1p_print("2p:  ", p2, stderr);
                g1p_print("2q:  ", q2, stderr);
            }
        }

        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}
 