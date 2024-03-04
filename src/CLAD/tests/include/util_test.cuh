#ifndef UTIL_TEST_CUH
#define UTIL_TEST_CUH


//Declaration shorthand
#define TESTFUN(X) extern "C"__global__ bool X(testval_t *testval, const size_t testsize)

//Controls printing
extern int verbosity;


#endif
