#ifndef FR_TEST_CUH
#define FR_TEST_CUH

#include "fr.cuh"
#include "util_test.cuh"

typedef fr_t testval_t;

//Constant tests
TESTFUN(FrTestKAT);
TESTFUN(FrTestFibonacci);

//Linear time tests

TESTFUN(FrTestCmp);
TESTFUN(FrTestMulConst);
TESTFUN(FrTestAdd);
TESTFUN(FrTestSub);
TESTFUN(FrTestSqr);
TESTFUN(FrTestMul);
TESTFUN(FrTestInv);
TESTFUN(FrTestMMA);

//Quadratic time tests
TESTFUN(FrTestSqr2);
TESTFUN(FrTestCommutativeAdd);
TESTFUN(FrTestCommutativeMul);

//Cubic time tests
TESTFUN(FrTestAssociativeAdd);
TESTFUN(FrTestAssociativeMul);
TESTFUN(FrTestAddDistributiveLeft);
TESTFUN(FrTestAddDistributiveRight);
TESTFUN(FrTestSubDistributiveLeft);
TESTFUN(FrTestSubDistributiveRight);



#endif
