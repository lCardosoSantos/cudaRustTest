#include "testUtil.cuh"

int verbosity;


//TODO: This function may not be necessary anymore. Or the function pointers can be pointers to instantiations of the test functions
// and then those pointers are visible to rust. Or those pointers can be in an array, and an enum is used to run the tests.
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
