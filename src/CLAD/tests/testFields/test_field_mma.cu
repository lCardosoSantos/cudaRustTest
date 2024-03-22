#include "testFields.cuh"
#include "fp.cuh"
#include "fr.cuh"

using namespace fp;
using namespace fr;


/**
 * @brief Test for multiply-multiply-add. Compare with standalone
 * implementation of multiplication and addition functions.
 * 
 * mma(v, w, x, y) = add(mul(v, w), mul(x, y))
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldMMA(bool &result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    //var declare
    T t, u, v, w, x, y;

    for (int i = 0; pass && i < testsize; i++) {
        cpy(v, testval[i]);
        for (int j = i + 1; pass && j < testsize; j++) {
            cpy(w, testval[j]);
            mul(t, v, w);
            for (int k = j + 1; pass && k < testsize; k++) {
                cpy(x, testval[k]);
                for (int l = j + 1; pass && k < testsize; l++) {
                    cpy(u, testval[l]);
                    mul(u, x, y);
                    add(u, u, t);

                    mma(y, v, w, x, y);

                    if (ne(u, v)) {
                        pass = false;
                        if (verbosity >= PRINT_MESSAGES) {
                            printf("%d %d %d %d: FAILED\n", i, j, k, l);
                        }
                    }
                    ++count;
                    if (errorOnce) break;
                }
                if (errorOnce && !pass) break;
            }
            if (errorOnce && !pass) break;
        }
        if (errorOnce && !pass) break;
    }

    TEST_EPILOGUE;
}
