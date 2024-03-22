#include <unistd.h>
#include <stdint.h>
#include "testUtil.cuh"

__managed__ enum verbosityLevel verbosity;
__managed__ bool stdout_isatty;
__managed__ bool errorOnce;


/**
 * @brief Initializes parameters for testing
 * 
 * @param verbosityLevel 
 */
template<typename T>
void init(const size_t testsize, T* testval, enum verbosityLevel vl, bool _errorOnce){
    verbosity = vl;
    stdout_isatty = isatty(fileno(stdout)); //Done here since isatty() cannot be called from the device.
    errorOnce = _errorOnce;

    //Allocate
    cudaError_t  err;
    err = cudaMallocManaged(testval, testsize * sizeof(T));
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
    
}


// //TODO: This function may not be necessary anymore. Or the function pointers can be pointers to instantiations of the test functions
// // and then those pointers are visible to rust. Or those pointers can be in an array, and an enum is used to run the tests.
// template<typename T>
// bool runTest( bool(*testfunc)(bool, T*, const size_t), 
//                          void *T, const size_t testsize,
//                          dim3 block, dim3 grid){

//     cudaError_t err;
//     bool result;

//     (*testfunc<T>)<<<grid, block>>>(result, testval, testsize); //TODO: Does this magic works?
//     CUDASYNC("");

//     return result;

//         //Encapsulates the testfunction kernel call so the tests
//         //can be called from Rust 
//         }



