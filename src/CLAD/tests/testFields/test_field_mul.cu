#include "testFields.cuh"
#include "fp.cuh"
#include "fr.cuh"

using namespace fp;
using namespace fr;


/**
 *  @brief Test self consistency in multiplication by constant:
 * 
 * 2(4x) = =8x
 * 2(2(2(2(2(2x))))) == 4(4(4x)) == 8(8x)
 * 3(4x) == 12(x)
 * 3(3(3(2(4(8x))))) == 12(12(12x))
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldMulConst(bool result, T *testval, const size_t testsize){
    #define ITER 100
    TEST_PROLOGUE;
     // 2*4 == 8

    for (int i=0; pass && i<testsize; i++) {
        T x2x4, x8;

        cpy(x2x4, testval[i]);
        cpy(x8,   testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            T x1;
            cpy(x1, x2x4);

            x2(x2x4, x2x4);
            x4(x2x4, x2x4);

            x8(x8, x8);

            if (fp_neq(x2x4, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                field_print("1   : ",  x1);
                field_print("2*4 : ",  x2x4);
                field_print("8   : ",  x8);
            }
            ++count; if (errorOnce && !pass) break;
        }
        if (errorOnce && !pass) break;
    }

    // 2*2*2*2*2*2 == 4*4*4 == 8*8

    for (int i=0; pass && i<testsize; i++) {
        T x2, x4, x8;

        cpy(x2, testval[i]);
        cpy(x4, testval[i]);
        cpy(x8, testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            T x1;
            cpy(x1, x2);

            x2(x2, x2);
            x2(x2, x2);
            x2(x2, x2);
            x2(x2, x2);
            x2(x2, x2);
            x2(x2, x2);

            x4(x4, x4);
            x4(x4, x4);
            x4(x4, x4);

            x8(x8, x8);
            x8(x8, x8);

            if (fp_neq(x2, x4) || fp_neq(x2, x8)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                field_print("1   : ",  x1);
                field_print("2^6 : ",  x2);
                field_print("4^3 : ",  x4);
                field_print("8^2 : ",  x8);
            }
            ++count; if (errorOnce && !pass) break;
        }
        if (errorOnce && !pass) break;
    }

    // 3*4 == 12

    for (int i=0; pass && i<testsize; i++) {
        T x3x4, x12;

        cpy(x3x4, testval[i]);
        cpy(x12,  testval[i]);

        for (int j=0; pass && j<ITER; j++) {
            T x1;
            cpy(x1, x3x4);

            fp_x3(x3x4, x3x4);
            x4(x3x4, x3x4);

            fp_x12(x12, x12);

            if (fp_neq(x3x4, x12)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", j);
                field_print("1   : ",  x1);
                field_print("3*4 : ",  x3x4);
                field_print("12  : ",  x12);
            }
            ++count; if (errorOnce && !pass) break;
        }
        if (errorOnce && !pass) break;
    }

    // 12+8 == 4(3+2)

    for (int i=0; pass && i<testsize; i++) {
        T x1, x2, x3, x8, x12, l, r;

        cpy(l, testval[i]);
        cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            cpy(x1, l);

            x2(x2, l);
            fp_x3(x3, l);
            x8(x8, l);
            fp_x12(x12, l);

            fp_add(l, x12, x8);

            fp_add(r, x3, x2);
            x4(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                field_print("1      : ",  x1);
                field_print("12+8   : ",  l);
                field_print("4(3+2) : ",  r);
            }
            ++count;if (errorOnce && !pass) break;
        }
        if (errorOnce && !pass) break;
    }

    // 3*3*3*2*4*8 == 12*12*12

    for (int i=0; pass && i<testsize; i++) {
        T x1, l, r;

        cpy(l, testval[i]);
        cpy(r, testval[i]);

        for (int j=0; pass && j<ITER; j++) {

            cpy(x1, l);

            fp_x3(l, l);
            fp_x3(l, l);
            fp_x3(l, l);
            x2(l, l);
            x4(l, l);
            x8(l, l);

            fp_x12(r, r);
            fp_x12(r, r);
            fp_x12(r, r);

            if (fp_neq(l, r)) {
                pass = false;

                printf("%d: FAILED: inconsistent result\n", i);
                field_print("1           : ",  x1);
                field_print("3*3*3*2*4*8 : ",  l);
                field_print("12*12*12    : ",  r);
            }
            ++count; if (errorOnce && !pass) break;
        }
    }

    TEST_EPILOGUE;
}

/**
 * @brief Multiplication test, using different values for different threads.
 * 
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldMul(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    //var declare
    T x, z;

    for (int i=0; pass && i<testsize; i++){
        x = fr_t(i);
        mul(z, x, x);

        if (z[0] !=     i*i) pass = false;
        if (z[1] !=       0) pass = false;
        if (z[2] !=       0) pass = false;
        if (z[3] !=       0) pass = false;

        if(!pass){
            if (verbosity >= PRINT_MESSAGES){
                printf("%d: FAILED\n", i);
                field_print("z    : ",  z);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for the commutative property of addition
 * 
 * x*y == y*x
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldCommutativeMul(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    //var declare
    T x, y;

    for (int i=0; pass && i<testsize; i++){
        for (int j=0; pass && j<testsize; j++){
            cpy(x, testval[i]);
            cpy(y, testval[j]);

            mul(x, x, testval[i]);
            mul(y, y, testval[j]);

            if(neq(x,y)){
                pass = false;
                if (verbosity >= PRINT_MESSAGES){
                    printf("%d: FAILED\n", i);
                    field_print("x    : ",  testval[i]);
                    field_print("y    : ",  testval[j]);    
                    field_print("x*y  : ",  x);
                    field_print("y*x  : ",  y);   
                }
            }
            ++count;
            if (errorOnce) break;
        }
        if (errorOnce && !pass) break;
    }

    TEST_EPILOGUE;
}

/**
 * @brief Test for the associative property of multiplication
 * 
 * (x*y)*z == x*(y*z)
 * 
 * @param testval 
 * @param testsize 
 * 
 * @return bool 
 */
template<typename T>
 __global__ void TestFieldAssociativeMul(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;

    // var declare
    T a, b, c;

    for (int i = 0; pass && i < testsize; i++) {
        for (int j = 0; pass && j < testsize; j++) {
            for (int k = 0; pass && j < testsize; k++) {

                cpy(a, testval[i]); // x
                cpy(b, testval[j]); // y
                cpy(c, testval[i]); // x

                mul(a, a, testval[j]); // x * y
                mul(a, a, testval[k]); // (x * y) * z

                mul(b, b, testval[k]); // y * z
                mul(c, c, b);          // x * (y * z)

                if (neq(a, c)) {
                    pass = false;
                    if (verbosity >= PRINT_MESSAGES) {
                        printf("%d: FAILED\n", i);
                        field_print("x    : ", testval[i]);
                        field_print("y    : ", testval[j]);
                        field_print("k    : ", testval[k]);
                        field_print("(x*y)*z = ", a);
                        field_print("x*(y*z) = ", c);
                    }
                }
                ++count;
                if (errorOnce)
                    break;
            }
            if (errorOnce && !pass)
                break;
        }

        if (errorOnce && !pass)
            break;
    }

    TEST_EPILOGUE;
}
