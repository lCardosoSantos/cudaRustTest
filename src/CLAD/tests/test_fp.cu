#include "testUtil.cuh"
#include "testFields.cuh"
#include "fp.cuh"

#define TESTSIZE (size_t)1024


// extern "C" void run_fp_tests(){
//     // setupTestCommon();
//     // //Constant tests
//     // TESTRUN(TestFieldKAT);
//     // // TESTRUN(TestFieldFibonacci);

//     // //Linear time tests

//     // TESTRUN(TestFieldCmp);
//     // TESTRUN(TestFieldMulConst);
//     // TESTRUN(TestFieldAdd);
//     // TESTRUN(TestFieldSub);
//     // // TESTRUN(TestFieldAddsub);//Not implemented
//     // TESTRUN(TestFieldSqr);
//     // TESTRUN(TestFieldMul);
//     // TESTRUN(TestFieldInv);
//     // TESTRUN(TestFieldMMA);

//     // //Quadratic time tests
//     // TESTRUN(TestFieldSqr2);
//     // TESTRUN(TestFieldCommutativeAdd);
//     // TESTRUN(TestFieldCommutativeMul);

//     // //Cubic time tests
//     // TESTRUN(TestFieldAssociativeAdd);
//     // TESTRUN(TestFieldAssociativeMul);
//     // TESTRUN(TestFieldAddDistributiveLeft);
//     // TESTRUN(TestFieldAddDistributiveRight);
//     // TESTRUN(TestFieldSubDistributiveLeft);
//     // TESTRUN(TestFieldSubDistributiveRight);

// }

__managed__  fp_t *testval_fp;
__managed__ bool res; 

extern "C" void run_fp_tests(){
    res=false;
    cudaError_t err;
    init(TESTSIZE, testval_fp);
    
    //Constant tests
    TESTMSG(TestFieldAdd);
    TestFieldAdd<<<1,1>>>(res, testval_fp, TESTSIZE);

    CUDASYNC("TestFieldAdd");
    
    printf("res = %d\n", res);

    cudaFree(testval_fp);
}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("Fp tests");
    run_fp_tests();
}

#endif
