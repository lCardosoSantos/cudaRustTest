#include "testUtil.cuh"
#include "testFields.cuh"
#include "fr.cuh"

#define TESTSIZE 1024

// extern "C" void run_fr_tests(){
//     setupTestCommon ();
//     //Constant tests
//     TESTRUN(TestFieldKAT);

//     //Linear time tests

//     TESTRUN(TestFieldCmp);
//     TESTRUN(TestFieldMulConst);
//     TESTRUN(TestFieldAdd);
//     TESTRUN(TestFieldSub);
//     // TESTRUN(TestFieldAddsub);//Not implemented
//     TESTRUN(TestFieldSqr);
//     TESTRUN(TestFieldMul);
//     TESTRUN(TestFieldInv);
//     TESTRUN(TestFieldMMA);

//     //Quadratic time tests
//     TESTRUN(TestFieldSqr2);
//     TESTRUN(TestFieldCommutativeAdd);
//     TESTRUN(TestFieldCommutativeMul);

//     //Cubic time tests
//     TESTRUN(TestFieldAssociativeAdd);
//     TESTRUN(TestFieldAssociativeMul);
//     TESTRUN(TestFieldAddDistributiveLeft);
//     TESTRUN(TestFieldAddDistributiveRight);
//     TESTRUN(TestFieldSubDistributiveLeft);
//     TESTRUN(TestFieldSubDistributiveRight);

// }

__managed__  fr_t *testval;


extern "C" void run_fr_tests(){
    bool res=false;
    init(TESTSIZE, testval);
    
    //Constant tests
    TESTMSG(TestFieldAdd);
    TestFieldAdd<<<1,1>>>(res, testval, TESTSIZE);
    printf("%B\n", res);

    cudaFree(testval);
}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("fr tests");
    run_fr_tests();
}

#endif
