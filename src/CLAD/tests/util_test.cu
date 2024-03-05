#include "util_test.cuh"

int verbosity;

extern "C" bool runTest( bool(*testfunc)(bool, void*, const size_t), 
                         void *testval, const size_t testsize,
                         dim3 block, dim3 grid){

    cudaError_t err;
    bool result;

    (*testfunc)<<<grid, block>>>(result, testval, testsize); //TODO: Does this magic works?
    CUDASYNC("");

    return result;

        //Encapsulates the testfunction kernel call so the tests
        //can be called from Rust 
        }
