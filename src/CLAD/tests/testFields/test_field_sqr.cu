#include "testFields.cuh"
#include "fp.cuh"
#include "fr.cuh"

using namespace fp;
using namespace fr;


/**
 * @brief Test for squaring on Fp. Checks for self consistency:
 * 
 * (x+n)^2 == x^2 + 2nx + n^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldSqr(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    const fp_t _1  = {1, 0, 0, 0, 0, 0}, 
               _2  = {2, 0, 0, 0, 0, 0}, 
               _4  = {4, 0, 0, 0, 0, 0}, 
               _6  = {6, 0, 0, 0, 0, 0},
               _16 = {16, 0, 0, 0, 0, 0}, 
               _36 = {36, 0, 0, 0, 0, 0};

    fp_t x, xsqr, x2, x4, x8, x12, l, r;

    // (x+n)^2 == x^2 + 2nx + n^2

    for (int i = 0; pass && i < testsize; i++) {

        cpy(x, testval[i]);

        sqr(xsqr, x);
        x2(x2, x);   // n = 1
        x4(x4, x);   // n = 2
        x8(x8, x);   // n = 4
        x12(x12, x); // n = 6

        // l = (x+1)^2
        add(l, x, _1);
        sqr(l, l);

        // r = x^2 + 2x + 1
        add(r, xsqr, x2);
        add(r, r, _1);

        if (neq(l, r)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES) {
                printf("%d: FAILED\n", i);
                field_print("x        : ", x);
                field_print("(x+1)^2  : ", l);
                field_print("x^2+2x+1 : ", r);
            }
            ++count;
            if (errorOnce)
                break;
        }

        // l = (x+2)^2
        add(l, x, _2);
        sqr(l, l);

        // r = x^2 + 4x + 4
        add(r, xsqr, x4);
        add(r, r, _4);

        if (neq(l, r)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES) {
                printf("%d: FAILED\n", i);
                field_print("x        : ", x);
                field_print("(x+2)^2  : ", l);
                field_print("x^2+4x+4 : ", r);
            }
        }
        ++count;
        if (errorOnce)
            break;

        // l = (x+4)^2
        add(l, x, _4);
        sqr(l, l);

        // r = x^2 + 8x + 16
        add(r, xsqr, x8);
        add(r, r, _16);

        if (neq(l, r)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES) {
                printf("%d: FAILED\n", i);
                field_print("x         : ", x);
                field_print("(x+4)^2   : ", l);
                field_print("x^2+8x+16 : ", r);
            }
        }
        ++count;
        if (errorOnce)
            break;

        // l = (x+6)^2
        add(l, x, _6);
        sqr(l, l);

        // r = x^2 + 12x + 36
        add(r, xsqr, x12);
        add(r, r, _36);

        if (neq(l, r)) {
            pass = false;
            if (verbosity >= PRINT_MESSAGES) {
                printf("%d: FAILED\n", i);
                field_print("x          : ", x);
                field_print("(x+6)^2    : ", l);
                field_print("x^2+12x+36 : ", r);
            }
        }
        ++count;
        if (errorOnce)
            break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for squaring on Fp. Checks for self consistency:
 * 
 * (x+y)^2 == x^2 + 2xy + y^2
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldSqr2(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    T x, xsqr, x2, y, l, r;

    // (x+y)^2 == x^2 + 2xy + y^2

    for (int i = 0; pass && i < TESTVALS; i++) {
        cpy(x, testval[i]);
        sqr(xsqr, x);
        x2(x2, x);

        for (int j = i; pass && j < TESTVALS; j++) {

            // l = (x+y)^2
            add(l, x, y);
            sqr(l, l);

            // r = x^2 + 2xy + y^2
            add(r, x2, y); // 2x+y
            mul(r, r, y);  // 2xy+y^2
            add(r, xsqr, r);

            if (neq(l, r)) {
                pass = false;
                if (verbosity >= PRINT_MESSAGES) {
                    printf("%d: FAILED\n", i);
                    field_print("x           : ", x);
                    field_print("y           : ", y);
                    field_print("(x+y)^2     : ", l);
                    field_print("x^2+2xy+y^2 : ", r);
                }
            }
            ++count;
            if (errorOnce)
                break;
        }
        ++count;
        if (errorOnce)
            break;
    }

    TEST_EPILOGUE;
}
