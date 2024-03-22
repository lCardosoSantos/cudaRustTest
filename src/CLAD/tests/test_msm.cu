#include "testUtil.cuh"
#include "testMSM.cuh"


extern "C" void run_msm_tests(){
    // setupTestCommon();
    // //Constant tests
    // TESTRUN(testMSM);
    printf("msm tests");
}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("MSM tests");
    run_msm_tests();
}

#endif
