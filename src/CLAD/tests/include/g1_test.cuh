#ifndef G1_TEST_CUH
#define G1_TEST_CUH

#include "g1.cuh"
#include "util_test.cuh"

typedef g1p_t testval_t;

TESTFUN(G1TestKAT);
TESTFUN(G1TestFibonacci);
TESTFUN(G1TestDbl);
TESTFUN(G1TestCmp);
TESTFUN(G1TestCopy);
TESTFUN(G1TestEqNeq);
TESTFUN(G1TestReflexivity);
TESTFUN(G1TestSymmetry);
TESTFUN(G1TestAdditiveIdentity);
TESTFUN(G1TestMultiplicativeIdentity);
TESTFUN(G1TestAdditiveInverse);
TESTFUN(G1TestMultiplicativeInverse);
TESTFUN(G1TestCommutativeAdd);
TESTFUN(G1TestCommutativeMul);
TESTFUN(G1TestAssociativeAdd);
TESTFUN(G1TestAssociativeMul);
TESTFUN(G1TestDistributiveLeft);
TESTFUN(G1TestDistributiveRight);
TESTFUN(G1TestDouble);
TESTFUN(G1TestSquare);


#endif
