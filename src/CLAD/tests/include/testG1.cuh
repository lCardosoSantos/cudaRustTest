#ifndef G1_TEST_CUH
#define G1_TEST_CUH

#include "g1.cuh"
#include "testUtil.cuh"

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


#include "../testG1/test_g1_add.impl"
#include "../testG1/test_g1_cmp.impl"
#include "../testG1/test_g1_fibonacci.impl"
#include "../testG1/test_g1_KAT.impl"
#include "../testG1/test_g1_mul.impl"
#include "../testG1/test_g1_sub.impl"

#endif
