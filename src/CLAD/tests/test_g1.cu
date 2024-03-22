#include "testUtil.cuh"
#include "testG1.cuh"
#include "g1.cuh"

extern "C" void run_g1_tests(){
    // setupTestCommon();


    // TESTRUN(G1TestKAT);
    // TESTRUN(G1TestFibonacci);

    // TESTRUN(G1TestAdd);
    // TESTRUN(G1TestSub);
    // TESTRUN(G1TestAddsub);
    // TESTRUN(G1TestDbl);
    // TESTRUN(G1TestMul);
    // TESTRUN(G1TestNeg);

    // TESTRUN(G1TestCpy);
    // TESTRUN(G1TestIsPoint);
    // TESTRUN(G1TestEq);
    // TESTRUN(G1TestNeq);
    // TESTRUN(G1TestIsInf);

}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("g1 tests");
    run_g1_tests();
}

#endif
