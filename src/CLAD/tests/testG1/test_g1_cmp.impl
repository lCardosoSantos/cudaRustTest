#include "testG1.cuh"

/**
 * @brief Test for point copy
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestCpy(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t a, b;

    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(a, testval[i]);
        memcpy(&b, &(testval[i]), sizeof(g1p_t));

        if(g1p_neq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                printf("a == b \n");
                g1p_print("a:   ", a);
                g1p_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for point validation
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestIsPoint(bool *result, testval_t *testval, const size_t testsize){
    //#warning Function not implemented: G1TestIsPoint
    //TODO: What is a good way to generate invalid points?
}

/**
 * @brief Test for point equality
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestEq(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t a, b;

    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(a, testval[i]);
        g1p_cpy(b, testval[i]);

        if(!g1p_eq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                printf("a == b \n");
                g1p_print("a:   ", a);
                g1p_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for point inequality
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestNeq(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t a, b;

    for (int i=0; pass && i<testsize; i++){
        g1p_cpy(a, testval[i]);
        g1p_dbl(b, a);

        if(!g1p_neq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                printf("a != b \n");
                g1p_print("a:   ", a);
                g1p_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for point at infinity
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
 __global__ void G1TestIsInf(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t a;
    g1p_inf(a);

    if(!g1p_isInf(a)){
        pass = false;
        if (verbosity >= PRINT_MESSAGES){
            printf("%d: FAILED\n", 0);
            printf("isInf(inf) = True \n");
        }

    }

    TEST_EPILOGUE;
}

