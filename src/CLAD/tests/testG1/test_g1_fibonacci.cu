#include "testG1.cuh"

/**
 * @brief Test addition and multiplication using a fibonacci sequence (cascading
 * data dependency)
 * 
 * @return void 
 */
__global__ void G1TestFibonacci(bool *result, testval_t *testval, const size_t testsize){
    TEST_PROLOGUE;

    g1p_t p, q, t;
    fr_t k, l; 

    g1p_inf(p); // p  = 0
    g1p_gen(q); // q  = G

    zero(k);
    one(l);

    for (int i=0; pass && i<100; i++){
        add(k,l);
        g1p_add(p, p, q); // p += q

        g1p_gen(t);
        g1p_mul(t,t,k); // kG

        if (g1p_neq(p, t)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                g1p_print("p:  ", p);
                g1p_print("t:  ", t);
            }
            if (errorOnce) break;
            ++count;
        }

        add(l, k);
        g1p_add(q, q, p);  // q += p

        g1p_gen(t);
        g1p_mul(t, t, l);  // lG

        if (g1p_neq(q, t)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                g1p_print("q:  ", q);
                g1p_print("t:  ", t);
            }
            if (errorOnce) break;
            ++count;
        }
    }

    TEST_EPILOGUE;
}

