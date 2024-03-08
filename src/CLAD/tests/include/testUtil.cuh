#ifndef UTIL_TEST_CUH
#define UTIL_TEST_CUH

#include <stdio.h>

//Declaration shorthand
#define TESTFUN(X)  __global__ void X(bool result, testval_t *testval, const size_t testsize)

#define TESTFUN_T(X) template<typename T>  __global__ void X(bool result, T *testval, const size_t testsize)

//Controls printing
extern int verbosity;

#ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        return false;}                                                                                                     
#endif


//TODo
#ifndef RUNTEST
    #define RUNTEST 
#endif //RUNTEST



#endif //UTIL_TEST_CUH
