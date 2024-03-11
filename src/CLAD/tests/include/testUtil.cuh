#ifndef UTIL_TEST_CUH
#define UTIL_TEST_CUH

#include <stdio.h>

//Declaration shorthand
#define TESTFUN(X)  __global__ void X(bool *result, testval_t *testval, const size_t testsize)

#define TESTFUN_T(X) template<typename T>  __global__ void X(bool result, T *testval, const size_t testsize)

//Controls printing
enum verbosityLevel{
    NO_PRINT,
    PRINT_MESSAGES,
    PRINT_MESSAGES_TIME
};


extern __managed__ enum verbosityLevel  verbosity;
extern __managed__ bool stdout_isatty;
extern __managed__ bool errorOnce;

#ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        return false;}                                                                                                     
#endif


#define TEST_PROLOGUE \
    bool pass = true;\
    size_t count = 0;\
    if (verbosity >= PRINT_MESSAGES)\
        fprintf(stderr, "> RUN %s\n", __func__);


#define TEST_EPILOGUE \
    if (count > 0 && verbosity >= PRINT_MESSAGES && !errorOnce){\
        fprintf(stderr, "%d of %d  tests failed\n", count, testsize);\
    }\
    bool result = pass;



//TODo
#ifndef RUNTEST
    #define RUNTEST 
#endif //RUNTEST



#endif //UTIL_TEST_CUH
