#include "testUtil.cuh"
#include "testFields.cuh"
#include "fp.cuh"


extern "C" void run_fp_tests(){
    // setupTestCommon();
    // //Constant tests
    // TESTRUN(TestFieldKAT);
    // // TESTRUN(TestFieldFibonacci);

    // //Linear time tests

    // TESTRUN(TestFieldCmp);
    // TESTRUN(TestFieldMulConst);
    // TESTRUN(TestFieldAdd);
    // TESTRUN(TestFieldSub);
    // // TESTRUN(TestFieldAddsub);//Not implemented
    // TESTRUN(TestFieldSqr);
    // TESTRUN(TestFieldMul);
    // TESTRUN(TestFieldInv);
    // TESTRUN(TestFieldMMA);

    // //Quadratic time tests
    // TESTRUN(TestFieldSqr2);
    // TESTRUN(TestFieldCommutativeAdd);
    // TESTRUN(TestFieldCommutativeMul);

    // //Cubic time tests
    // TESTRUN(TestFieldAssociativeAdd);
    // TESTRUN(TestFieldAssociativeMul);
    // TESTRUN(TestFieldAddDistributiveLeft);
    // TESTRUN(TestFieldAddDistributiveRight);
    // TESTRUN(TestFieldSubDistributiveLeft);
    // TESTRUN(TestFieldSubDistributiveRight);

}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("Fp tests");
    run_fp_tests();
}

#endif
