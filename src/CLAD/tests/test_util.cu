// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include <unistd.h>
#include <stdint.h>
#include "testUtil.cuh"

__managed__ enum verbosityLevel verbosity;
__managed__ bool stdout_isatty;
__managed__ bool errorOnce;
__managed__ bool pass; 



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



