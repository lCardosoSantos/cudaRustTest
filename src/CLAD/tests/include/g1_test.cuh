#ifndef G1_TEST_CUH
#define G1_TEST_CUH

#include "g1.cuh"
#include "util_test.cuh"

typedef g1p_t testval_t;

TESTFUN(G1TestKAT);
TESTFUN(G1TestFibonacci);

TESTFUN(G1TestAdd);
TESTFUN(G1TestSub);
TESTFUN(G1TestAddsub);
TESTFUN(G1TestDbl);
TESTFUN(G1TestMul);
TESTFUN(G1TestNeg);

TESTFUN(G1TestCpy);
TESTFUN(G1TestIsPoint);
TESTFUN(G1TestEq);
TESTFUN(G1TestNeq);
TESTFUN(G1TestIsInf);

//TODO: Self consistency tests
#endif
