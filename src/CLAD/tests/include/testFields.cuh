#ifndef FP_TEST_CUH
#define FP_TEST_CUH

#include "testUtil.cuh"

//Constant tests
TESTFUN_T(TestFieldKAT);
// TESTFUN_T(TestFieldFibonacci);

//Linear time tests

TESTFUN_T(TestFieldCmp);
TESTFUN_T(TestFieldMulConst);
TESTFUN_T(TestFieldAdd);
TESTFUN_T(TestFieldSub);
// TESTFUN_T(TestFieldAddsub);//Not implemented
TESTFUN_T(TestFieldSqr);
TESTFUN_T(TestFieldMul);
TESTFUN_T(TestFieldInv);
TESTFUN_T(TestFieldMMA);

//Quadratic time tests
TESTFUN_T(TestFieldSqr2);
TESTFUN_T(TestFieldCommutativeAdd);
TESTFUN_T(TestFieldCommutativeMul);

//Cubic time tests
TESTFUN_T(TestFieldAssociativeAdd);
TESTFUN_T(TestFieldAssociativeMul);
TESTFUN_T(TestFieldAddDistributiveLeft);
TESTFUN_T(TestFieldAddDistributiveRight);
TESTFUN_T(TestFieldSubDistributiveLeft);
TESTFUN_T(TestFieldSubDistributiveRight);

#endif
