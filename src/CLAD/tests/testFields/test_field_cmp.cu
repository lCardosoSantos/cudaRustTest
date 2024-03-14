#include "testFields.cuh"
#include "fp.cuh"
#include "fr.cuh"

using namespace fp;
using namespace fr;


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
 __global__ void TestFieldCmp(bool result, T *testval, const size_t testsize){
    TEST_PROLOGUE;
    
    for (int i=0; pass && i<testsize; i++){
        cpy(a, testval[i]);
        memcpy(&b, &(testval[i]), sizeof(T));
        T c = {-1, -1, -1, -1};

        if(eq(a, b) == neq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "eq(x,x) != neq(x,x) \n", i);
                field_print("a:   ", a);
                field_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;

        if(neq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "neq(x,x)==false \n", i);
                field_print("a:   ", a);
                field_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;

        if(!neq(a, c)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "neq(x,y) == true \n", i);
                field_print("a:   ", a);
                field_print("c:   ", c);
            }
        }
        ++count;
        if (errorOnce) break;

        if(!eq(a, b)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "eq(x,x) == true \n", i);
                field_print("a:   ", a);
                field_print("b:   ", b);
            }
        }
        ++count;
        if (errorOnce) break;

        if(eq(a, c)){
            pass = false;
            if (verbosity >= PRINT_MESSAGES){
                fprintf(stderr, "%d: FAILED\n", i);
                fprintf(stderr, "eq(x,y) == false \n", i);
                field_print("a:   ", a);
                field_print("c:   ", c);
            }
        }
        ++count;
        if (errorOnce) break;
    }

    TEST_EPILOGUE;    
}
