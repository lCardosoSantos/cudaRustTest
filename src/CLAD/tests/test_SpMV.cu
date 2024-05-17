// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos

#include "SpMV.cuh"

#include <stdio.h>
#include <stdint.h>
#include <chrono>
#include <vector>

using namespace std;
using std::vector; 
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

#ifndef CUDASYNC
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess){                                                                                           \
    printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
    }                                                                                                     
#endif

#define CPU_TIMER_INIT time_point<Clock> start, end
#define CPU_TIMER_START start = Clock::now()
#define CPU_TIMER_END end = Clock::now()
#define CPU_TIMER_PRINT(msg) printf("%s : %ld ms\n", msg,  duration_cast<milliseconds>(end-start).count());


template<typename T, typename S>
void genRandomCSR(T *data, S *colidx, S *rowptr, S nCols, S nRows, S NNZ){
    //considers that data, col and rowPtr have been allocated with the correct dimensions.
    
    // Fill row_ptrs with 0 initially
    for (size_t i = 0; i < nRows+1; ++i) {
        rowptr[i] = 0;
    }

    // Randomly distribute nonzeros across the rows
    for (size_t i = 0; i < NNZ; ++i) {
        size_t row = rand() % nRows;
        rowptr[row + 1]++;
    }

    // Convert the count into actual indices
    for (size_t i = 1; i <= nRows; ++i) {
        rowptr[i] += rowptr[i - 1];
    }

    // Assign random columns and values to each entry
    for (size_t i = 0; i < NNZ; ++i) {
        colidx[i] = rand() % nCols;
        data[i] = rand();
    }
}

template<typename T, typename S>
void genRandomWitness(T &data, S len){
    //considers that data has been alocated to correct size
    for(size_t i = 0; i<len; i++){
        data[i] = rand();
    }
}

template<typename T>
vector<T> multiplyWitnessCPU_CSR(T *wit, T *data, size_t *colidx, size_t *indptr, 
                                         size_t nCols, size_t nRows, size_t NZZ){
    //runs the alg using CPU, on a canonical CSR
    
    vector<T> resCPU(nRows, 0);

    for(size_t i=0; i<nRows; i++){
        size_t start = indptr[i];
        size_t end   = indptr[i+1];
        for(size_t j=start; j < end; j++){
            T m = data[j]*wit[colidx[j]];
            resCPU[i] += m;
        }
    }
    return resCPU;
}

template<typename T>
vector<T> multiplyWitnessCPU_CSC(T *wit, T *data, size_t *rowidx, size_t *colptr, 
                                         size_t nCols, size_t nRows, size_t NZZ){
    //runs the alg using CPU, on a canonical CSC
    
    vector<T> resCPU(nRows, 0);

    for(size_t i=0; i<nCols; i++){
        size_t start = colptr[i];
        size_t end   = colptr[i+1];
        for(size_t j=start; j < end; j++){
            T m = data[j]*wit[i];
            resCPU[rowidx[j]] += m;
        }
    }
    return resCPU;
}

//initialized in SpMV.cu
extern __managed__ fr_t *cuda_data, *cuda_tmp_mul;
extern __managed__ fr_t **MultiplicationDestinationPointer;
extern __managed__ size_t *cuda_colptr, *cuda_rowidx, *cuda_rowptr;
extern __managed__ size_t cuda_NNZ, cuda_nRows, cuda_nCols;

__managed__ fr_t *witness, *resCuda;


extern "C" void run_matrixMult_tests(){
    printf("\nSparse Matrix Mult tests\n");

    #warning "Compiling for ~7GB data in this test"
    CPU_TIMER_INIT;
    static size_t aNNZ   = 104675466;
    static size_t bNNZ   = 57990791;
    static size_t cNNZ   = 9101902;
    static size_t anRows = 9825045;
    static size_t bnRows = 9825045;
    static size_t cnRows = 9825045;
    static size_t ncols  = 7999846;

    // static size_t aNNZ   = 100000;
    // static size_t bNNZ   = 1000;
    // static size_t cNNZ   = 1000;
    // static size_t anRows = 3000;
    // static size_t bnRows = 2000;
    // static size_t cnRows = 1000;
    // static size_t ncols  = 7000;

    fr_t *adata, *bdata, *cdata;
    size_t *acolidx, *bcolidx, *ccolidx,
           *arowptr, *browptr, *crowptr;

    adata = (fr_t*)malloc(aNNZ*sizeof(fr_t));
    bdata = (fr_t*)malloc(bNNZ*sizeof(fr_t));
    cdata = (fr_t*)malloc(cNNZ*sizeof(fr_t));

    acolidx = (size_t*)malloc(aNNZ*sizeof(size_t));   
    bcolidx = (size_t*)malloc(bNNZ*sizeof(size_t));
    ccolidx = (size_t*)malloc(cNNZ*sizeof(size_t));

    arowptr = (size_t*)malloc(anRows*sizeof(size_t));   
    browptr = (size_t*)malloc(bnRows*sizeof(size_t));
    crowptr = (size_t*)malloc(cnRows*sizeof(size_t));


    genRandomCSR<fr_t, size_t>(adata, acolidx, arowptr, ncols, anRows, aNNZ);
    genRandomCSR<fr_t, size_t>(bdata, bcolidx, browptr, ncols, bnRows, bNNZ);
    genRandomCSR<fr_t, size_t>(cdata, ccolidx, crowptr, ncols, cnRows, cNNZ);

    CPU_TIMER_START;
    sparseMatrixLoadCUDA(adata, acolidx, arowptr, aNNZ, anRows, 
                        bdata, bcolidx, browptr, bNNZ, bnRows, 
                        cdata, ccolidx, crowptr, cNNZ, cnRows, 
                        ncols);
    CPU_TIMER_END;
    CPU_TIMER_PRINT("Matrix Load:")

    cudaMallocManaged(&witness, ncols*sizeof(fr_t));    
    // for(int i=0; i<ncols; i++) witness[i]=fr_t(1); //witness[i]=fr_t(i); //easyWitness for debug
    genRandomWitness(witness, ncols);
    
    cudaMallocManaged(&resCuda, cuda_nRows*sizeof(fr_t));

    // no CPU fr implementation available
        // CPU_TIMER_START;
        // auto resCPU = multiplyWitnessCPU_CSC(witness, cuda_data, cuda_rowidx, cuda_colptr, ncols, abcnrows, abcNNZ);
        // CPU_TIMER_END;
        // CPU_TIMER_PRINT("CPU matrixMult:")


        // printf("A:\n");
        // printCSRMatrix(adata, acolidx, arowptr, anRows, ncols);
        // printf("B:\n");
        // printCSRMatrix(bdata, bcolidx, browptr, bnRows, ncols);
        // printf("C:\n");
        // printCSRMatrix(cdata, ccolidx, crowptr, cnRows, ncols);

        // printf("ABC:\n");
        // printCSCMatrix(cuda_data, cuda_rowidx, cuda_colptr, cuda_nRows, cuda_nCols);
        // printf("\n\n");

        // printf("resCPU:\n");
        // for(int i=0; i<abcnrows; i++) printf("%lu, ", resCPU[i]._[0]);
        // printf("\n\n");


    //force data to move to device from managed
    multiplyWitnessCUDA(resCuda, witness);
    for(int i=0; i<cuda_nRows; i++) resCuda[i]=fr_t(i); //witness[i]=fr_t(i);


    CPU_TIMER_START;
    multiplyWitnessCUDA(resCuda, witness);
    CPU_TIMER_END;
    CPU_TIMER_PRINT("GPU SpMV:")

    //No Host fr functions to compare.

        // int errConter = 0; 
        // for (int i = 0; i < cuda_nRows; i++) {
        //     if (fr_ne(resCPU[i], resCuda[i])) {
        //         printf("error at idx %d. Further errors supressed.\n", i);
        //         field_printh("cpu:  \n", resCPU[i]);
        //         field_printh("cuda: \n", resCuda[i]);
        //         errConter++;
        //         if (errConter > 10)break;
        //     }
        // }


}

#ifndef RUST_TEST 
int main(){
    run_matrixMult_tests();
}
#endif
