// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
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

#include "../testFields/test_field_add.impl" 
#include "../testFields/test_field_cmp.impl" 
#include "../testFields/test_field_inv.impl" 
#include "../testFields/test_field_KAT.impl" 
#include "../testFields/test_field_mma.impl" 
#include "../testFields/test_field_mul.impl" 
#include "../testFields/test_field_sqr.impl" 
#include "../testFields/test_field_sub.impl" 


#endif
