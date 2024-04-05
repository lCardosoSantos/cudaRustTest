#pragma once

#include <stdio.h>
#include <unistd.h>
#include <stdint.h>

#define COLOR_RED "\x1b[31m"
#define COLOR_GREEN "\x1b[32m"
#define COLOR_BLUE "\x1b[34m"
#define COLOR_RESET "\x1b[0m"
#define COLOR_BOLD "\x1b[1m"

//Controls printing
enum verbosityLevel{
    NO_PRINT,
    PRINT_MESSAGES,
    PRINT_MESSAGES_TIME
};

extern __managed__ enum verbosityLevel  verbosity;
extern __managed__ bool stdout_isatty;
extern __managed__ bool errorOnce;

//Declaration shorthand
#define TESTFUN(X)  __global__ void X(bool *result, testval_t *testval, const size_t testsize)

#define TESTFUN_T(X) template<typename T>  __global__ void X(bool &result, T *testval, const size_t testsize)

//TODO: Define test run macro
#define TESTMSG(X) if (stdout_isatty){printf(COLOR_BLUE "run " X COLOR_RESET " "); }\
                   else{ printf("run " X " "); }

//Print test result
#define PRINTPASS(pass) if (stdout_isatty){                                                             \
    printf("--- %s\n", pass ? COLOR_GREEN "PASS" COLOR_RESET : COLOR_RED COLOR_BOLD "FAIL" COLOR_RESET);\
    }else{                                        \
    printf("--- %s\n", pass ?  "PASS" :  "FAIL" );\
    }

//shortcut for running the tests
#define TEST_RUN(f, pass_var, testval_var, testsize_var) \
    f<<<1,1>>>(pass_var, testval_var, testsize_var);\
    CUDASYNC( #f );\
    if(err != cudaSuccess) pass_var = false; \
    PRINTPASS(pass_var)

#define DEBUGPRINT(var) printf(">>>> " #var " 0x%x \n", var);


#ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        }                                                                                                     
#endif


#define TEST_PROLOGUE \
    bool pass = true;\
    size_t count = 0;\
    if (verbosity >= PRINT_MESSAGES){\
        if (stdout_isatty){\
            printf(COLOR_BLUE "run %s" COLOR_RESET " ", __func__); \
            }\
        else{ printf("run %s ", __func__); }}
        // printf("> RUN %s\n", __func__);


#define TEST_EPILOGUE \
    if (count > 0 && verbosity >= PRINT_MESSAGES && !errorOnce){\
        printf("%u of %u  tests failed\n", count, testsize);\
    }\
    result = pass;

/**
 * @brief Initializes parameters for testing
 * 
 * @param verbosityLevel 
 */
template<typename T>
void init(const size_t testsize, T* testval, enum verbosityLevel vl = PRINT_MESSAGES_TIME, bool _errorOnce = true){
    verbosity = vl;
    stdout_isatty = isatty(fileno(stdout)); //Done here since isatty() cannot be called from the device.
    errorOnce = _errorOnce;

    //Allocate
    cudaError_t  err;
    err = cudaMallocManaged(&testval, testsize * sizeof(T));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    int i = 0;
    testval[i++].set_zero();
    testval[i++].set_one();

    for(int j=0; j<64; j++) testval[i++] = T(0, 0, 0, 1ull << j);
    for(int j=0; j<64; j++) testval[i++] = T(0, 0, 1ull << j);
    for(int j=0; j<64; j++) testval[i++] = T(0, 1ull << j);
    for(int j=0; j<64; j++) testval[i++] = T(1ull << j);

    FILE *urandom = fopen("/dev/urandom", "r");

    if (!urandom){
        printf("WARNING: Nonfatal: unable to access /dev/urandom. \n");
        return;
    }

    for(; i<testsize; i++){
        uint64_t tmp[4];
        fread(tmp, sizeof(uint64_t), 4, urandom);
        testval[i] = T(tmp[0], tmp[1], tmp[2], tmp[3]);
    }

    if (!urandom) fclose(urandom);
    
    printf(">INIT %d\n", i);
}

template<typename T>
bool runTest( bool(*testfunc)(bool, T*, const size_t), 
                         T *testval, const size_t testsize,
                         dim3 block, dim3 grid);
