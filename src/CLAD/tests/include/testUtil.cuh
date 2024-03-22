#pragma once

#include <stdio.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_RESET "\x1b[0m"
#define COLOR_BOLD "\x1b[1m"

//Declaration shorthand
#define TESTFUN(X)  __global__ void X(bool *result, testval_t *testval, const size_t testsize)

#define TESTFUN_T(X) template<typename T>  __global__ void X(bool &result, T *testval, const size_t testsize)

//TODO: Define test run macro
#define TESTMSG(X) printf(COLOR_BOLD COLOR_BLUE "run test" #X COLOR_RESET "\n")


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
        printf("> RUN %s\n", __func__);


#define TEST_EPILOGUE \
    if (count > 0 && verbosity >= PRINT_MESSAGES && !errorOnce){\
        printf("%u of %u  tests failed\n", count, testsize);\
    }

//initializes variables for testing.
template<typename T>
void init(const size_t testsize, T* testval, enum verbosityLevel vl=PRINT_MESSAGES_TIME, bool _errorOnce=true);

template<typename T>
bool runTest( bool(*testfunc)(bool, T*, const size_t), 
                         T *testval, const size_t testsize,
                         dim3 block, dim3 grid);
