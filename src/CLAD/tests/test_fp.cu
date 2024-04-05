#include "testUtil.cuh"
#include "testFields.cuh"
#include "fp.cuh"

#define TESTSIZE (size_t)256

__managed__  fp_t *testval_fp;

extern "C" void run_fp_tests(){\
    printf("\nFp tests\n");

    pass=false;
    cudaError_t err;
    init(TESTSIZE, testval_fp);

    //commented tests are not implemented funcs

    //Linear time tests
    TEST_RUN(TestFieldCmp, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldMulConst, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldAdd, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldSub, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldMul, pass, testval_fp, TESTSIZE);

        // TEST_RUN(TestFieldSqr, pass, testval_fp, TESTSIZE);
        // TEST_RUN(TestFieldInv, pass, testval_fp, TESTSIZE);
        // TEST_RUN(TestFieldMMA, pass, testval_fp, TESTSIZE); 


    //Quadratic time tests
        // TEST_RUN(TestFieldSqr2, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldCommutativeAdd, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldCommutativeMul, pass, testval_fp, TESTSIZE);

    //Cubic time tests
    TEST_RUN(TestFieldAssociativeAdd, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldAssociativeMul, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldAddDistributiveLeft, pass, testval_fp, TESTSIZE);
    TEST_RUN(TestFieldAddDistributiveRight, pass, testval_fp, TESTSIZE);
        // TEST_RUN(TestFieldSubDistributiveLeft, pass, testval_fp, TESTSIZE);
        // TEST_RUN(TestFieldSubDistributiveRight, pass, testval_fp, TESTSIZE);

    printf("\n---\n");
    cudaFree(testval_fp);
}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("Fp tests\n");
    run_fp_tests();
}

#endif
