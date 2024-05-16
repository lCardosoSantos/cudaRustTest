// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include "testUtil.cuh"
#include "testFields.cuh"
#include "fr.cuh"

#define TESTSIZE (size_t)256

__managed__  fr_t *testval_fr;

extern "C" void run_fr_tests(){
    printf("\nFr tests\n");

    pass=false;
    cudaError_t err;
    init(TESTSIZE, testval_fr);

    //Linear time tests
    TEST_RUN(TestFieldCmp, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldMulConst, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldAdd, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldSub, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldMul, pass, testval_fr, TESTSIZE);

    //Quadratic time tests
    TEST_RUN(TestFieldCommutativeAdd, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldCommutativeMul, pass, testval_fr, TESTSIZE);

    //Cubic time tests
    TEST_RUN(TestFieldAssociativeAdd, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldAssociativeMul, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldAddDistributiveLeft, pass, testval_fr, TESTSIZE);
    TEST_RUN(TestFieldAddDistributiveRight, pass, testval_fr, TESTSIZE);

    printf("\n---\n");
    cudaFree(testval_fr);

}

//Defined if the file is compiled for the Rust Library.
#ifndef RUST_TEST 

int main(int argc, char **argv){
    printf("fr tests\n");
    run_fr_tests();
}

#endif
