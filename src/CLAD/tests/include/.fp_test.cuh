#ifndef FP_TEST_CUH
#define FP_TEST_CUH

#include "fp.cuh"
#include "util_test.cuh"

typedef fp_t testval_t;

//Constant tests
TESTFUN(FpTestKAT);
TESTFUN(FpTestFibonacci);

//Linear time tests

TESTFUN(FpTestCmp);
TESTFUN(FpTestMulConst);
TESTFUN(FpTestAdd);
TESTFUN(FpTestSub);
TESTFUN(FpTestAddsub);
TESTFUN(FpTestSqr);
TESTFUN(FpTestMul);
TESTFUN(FpTestInv);
TESTFUN(FpTestMMA);

//Quadratic time tests
TESTFUN(FpTestSqr2);
TESTFUN(FpTestCommutativeAdd);
TESTFUN(FpTestCommutativeMul);

//Cubic time tests
TESTFUN(FpTestAssociativeAdd);
TESTFUN(FpTestAssociativeMul);
TESTFUN(FpTestAddDistributiveLeft);
TESTFUN(FpTestAddDistributiveRight);
TESTFUN(FpTestSubDistributiveLeft);
TESTFUN(FpTestSubDistributiveRight);

#endif
