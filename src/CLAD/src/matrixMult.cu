// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include <stdio.h>
#include <stdint.h>
#include "fr.cuh"

#ifndef CUDASYNC
#define CUDASYNC(fmt, ...)                                                                                             \
    err = cudaDeviceSynchronize();                                                                                     \
    if (err != cudaSuccess){                                                                                           \
    printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
    }                                                                                                     
#endif

#define CUDA_CHECK_ALLOCATION \
    if( err != cudaSuccess ){ \
            printf("FATAL: Cuda memory allocation failed!\n"); \
            printf("Error: %d: %s\n", err, cudaGetErrorName(err)); \
            }

// #define CPU_TIMER_INIT time_point<Clock> start, end
// #define CPU_TIMER_START start = Clock::now()
// #define CPU_TIMER_END end = Clock::now()
// #define CPU_TIMER_PRINT(msg) printf("%s : %ld ms\n", msg,  duration_cast<milliseconds>(end-start).count());


__managed__ fr_t *cuda_data, *cuda_tmp_mul;
__managed__ fr_t **MultiplicationDestinationPointer;
__managed__ size_t *cuda_colptr, *cuda_rowidx, *cuda_rowptr; //TODO: I have an intuition that rowIdx is not necessary, only rowPtr. return here after kernel is programmed
__managed__ size_t cuda_NNZ, cuda_nRows, cuda_nCols;
size_t ANROWS, BNROWS, CNROWS; //need to keep track for separating the witness pointers

bool cuda_matrix_multiplication_ready = false;

template<typename T, typename S>
void sparseMatrixLoad(const T *A_data, const S *A_colidx, const S *A_indptr, const S A_NNZ, const S A_nRows, 
                      const T *B_data, const S *B_colidx, const S *B_indptr, const S B_NNZ, const S B_nRows, 
                      const T *C_data, const S *C_colidx, const S *C_indptr, const S C_NNZ, const S C_nRows, 
                      const S nCols_l){

    cudaError_t err;

    // managed memory allocation
    cuda_NNZ = A_NNZ+B_NNZ+C_NNZ;
    cuda_nRows = A_nRows + B_nRows + C_nRows;
    cuda_nCols = nCols_l;
    ANROWS = A_nRows; BNROWS = B_nRows; CNROWS = C_nRows;

    err = cudaMallocManaged(&cuda_data,    cuda_NNZ * sizeof(T)); CUDA_CHECK_ALLOCATION;
    err = cudaMallocManaged(&cuda_tmp_mul, cuda_NNZ * sizeof(T)); CUDA_CHECK_ALLOCATION;

    err = cudaMallocManaged(&MultiplicationDestinationPointer, cuda_NNZ * sizeof(T *)); CUDA_CHECK_ALLOCATION;

    err = cudaMallocManaged(&cuda_colptr, (cuda_nCols+1) * sizeof(S)); CUDA_CHECK_ALLOCATION;
    err = cudaMallocManaged(&cuda_rowptr, (cuda_nRows+1) * sizeof(S)); CUDA_CHECK_ALLOCATION;
    err = cudaMallocManaged(&cuda_rowidx, (cuda_NNZ+1) * sizeof(S)); CUDA_CHECK_ALLOCATION;

    

    // csr - csc convert

    //// initialize column pointers to zero
    for(size_t i = 0; i <= cuda_nCols; i++){
        cuda_colptr[i] = 0;
    }

    //// count the number of NNZ on each column
    for(size_t i = 0; i < A_NNZ; i++)
        cuda_colptr[A_colidx[i] + 1]++;
    for(size_t i = 0; i < B_NNZ; i++)
        cuda_colptr[B_colidx[i] + 1]++;
    for(size_t i = 0; i < C_NNZ; i++)
        cuda_colptr[C_colidx[i] + 1]++;

    //// convert the column counter into column pointer by commulative add
    for(size_t i = 0; i<cuda_nCols; i++)
        cuda_colptr[i+1] += cuda_colptr[i];

    //// init temporary array to track columns
    size_t *tmp = (size_t *)malloc(cuda_nCols * sizeof(size_t));
    for(size_t i = 0; i < cuda_nCols; i++)
        tmp[i] = cuda_colptr[i];

    //// Fill data and precomputer destination pointers (note, cuda processors do not have the same pointer arithmetic instructions that x86 does, so by using a bit more RAM, we can avoid pointer arithmethic in each multiplication)
    for(size_t i=0; i<A_nRows; i++){
        for(size_t j=A_indptr[i]; j<A_indptr[i+1]; j++){
            size_t col = A_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i;
            cuda_data[dest] = A_data[j]; //TODO: Use fromMont in case source data is not in montgomery format
            MultiplicationDestinationPointer[dest] = &(cuda_tmp_mul[j]);
            tmp[col]++;
        }
    }

    for(size_t i=0; i<B_nRows; i++){
        for(size_t j=B_indptr[i]; j<B_indptr[i+1]; j++){
            size_t col = B_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i+A_nRows;
            cuda_data[dest] = B_data[j]; //TODO: Use fromMont in case source data is not in montgomery format
            tmp[col]++;
            MultiplicationDestinationPointer[dest] = &(cuda_tmp_mul[j + A_NNZ]);
        }
    }

    for(size_t i=0; i<C_nRows; i++){
        for(size_t j=C_indptr[i]; j<C_indptr[i+1]; j++){
            size_t col = C_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i+A_nRows+B_nRows;
            cuda_data[dest] = C_data[j]; //TODO: Use fromMont in case source data is not in montgomery format
            tmp[col]++;
            MultiplicationDestinationPointer[dest] = &(cuda_tmp_mul[j + A_NNZ + B_NNZ]);
        }
    }

    //// update rowptr

    for(size_t i=0; i<=A_nRows; i++)
        cuda_rowptr[i] = A_indptr[i];

    for(size_t i=0; i<=B_nRows; i++)
        cuda_rowptr[i+A_nRows] = B_indptr[i] + A_NNZ;

    for(size_t i=0; i<=C_nRows; i++)
        cuda_rowptr[i+A_nRows+B_nRows] = C_indptr[i] + A_NNZ + B_NNZ;

    //// free temporary
    free(tmp);

    //// set load flag
    cuda_matrix_multiplication_ready = true;
}

template<typename T>
__global__  void multiplyWitnessKernel(T *witness){
    //Result is kept in pre-allocated managed memory (~5,5 GB)
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    // unsigned gdim = gridDim.x;
    extern __shared__ uint32_t shareptr[];

    uint32_t *W = (shareptr); //reserve start of shared memory for witness
    uint32_t *M = (shareptr + 8); //one extra padding bank to avoid load conflict
    T *wit = (T*)W; 

    size_t start = cuda_colptr[bidx];
    size_t end = cuda_colptr[bidx+1];
    size_t mend = end - ((end-start)%bdim);
    
    if(start==end) return; //empty column

    //Load witness[col] into shared memory and syncThreads
    uint32_t *basePointer = (uint32_t *)(&witness[bidx]);
    if(tidx < 8){
        W[tidx] = basePointer[tidx];
    } 
    __syncthreads();
    //loop over column data,loading one block into shared memory up to a multiple of block size
    for(size_t i=start; i<mend; i+=bdim){
        //load data into shared memory
        uint32_t *mpointer = (uint32_t *)(&cuda_data[i+tidx]);
        for(int j=0; j<8; j++){
            M[(10*tidx)+j] = mpointer[j];
        }
        __syncthreads();

        //multiply and write to destination
        T *sharedM = (T*)(&M[10*tidx]);
        fr_mul(*MultiplicationDestinationPointer[i+tidx], *sharedM, *wit);

        // if(bidx==0) printf(">>> [b0t%02d]row %d written to adress %p\n", tidx, cuda_rowidx[i+tidx], MultiplicationDestinationPointer[+tidx]);

    }
    
    //load data into shared memory
    if (tidx < end-mend){
        uint32_t *mpointer = (uint32_t *)(&cuda_data[mend+tidx]);
        for(int j=0; j<8; j++){
            M[(10*tidx)+j] = mpointer[j];
        }
    }
    __syncthreads();

    //multiplication tail
    if (tidx < end-mend){
        //multiply and write to destination
        T *sharedM = (T*)(&M[10*tidx]);
        fr_mul(*MultiplicationDestinationPointer[mend+tidx], *sharedM, *wit);
    }
}

template<typename T>
__global__  void sumWitnessKernel(T *res){
    //Input is in pre-allocated managed memory
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    // unsigned gdim = gridDim.x; 

    //each block operates on a single row
    extern __shared__ uint32_t shareptr[];
    
    //init local accumulator to zero, keep in register
    T accumulator = fr_t(0);

    //loop over row data, loading one blok into shared memory up to a multiple of block size
    size_t start = cuda_rowptr[bidx];
    size_t end = cuda_rowptr[bidx+1];
    size_t mend = end - ((end-start)%bdim);

    if(start==end){//empty row
        res[bidx] = accumulator;
        // if (tidx==0)printf("Block %d early termination\n", bidx);
        return; 
    }
    
    //loop over row data,loading one block into shared memory up to a multiple of block size
    for(size_t i=start; i<mend; i+=bdim){
        //load data into shared memory
        uint32_t *mpointer = (uint32_t *)(&cuda_tmp_mul[i+tidx]);
        for(int j=0; j<8; j++){
            shareptr[(10*tidx)+j] = mpointer[j];
        }
        __syncthreads();

        //accumulate
        T *sharedM = (T*)(&shareptr[10*tidx]);
        fr_add(accumulator, accumulator, *sharedM);
    }

    // if (tidx == 0) printf("block %02d end main loop\n", bidx);

    //load and accumulate tail

    //load data into shared memory
    if (tidx < end-mend){
        uint32_t *mpointer = (uint32_t *)(&cuda_tmp_mul[mend+tidx]);
        for(int j=0; j<8; j++){
            shareptr[(10*tidx)+j] = mpointer[j];
        }
    }
    __syncthreads();

    if (tidx < end-mend){
        //accumulate
        T *sharedM = (T*)(&shareptr[10*tidx]);
        fr_add(accumulator, accumulator, *sharedM);
    }

    // if (tidx == 0) printf("block %02d end tail sum loop\n", bidx);

    //reduce sum
    T *acc = (T *)(&shareptr[10*tidx]);
    *acc = accumulator;
    __syncthreads();
    for (int stride = bdim / 2; stride > 0; stride>>=1){
        if (tidx < stride){
            // *acc += (T *)(&shareptr[10*(tidx+stride)]);
            fr_add(*acc, *acc, *(T *)(&shareptr[10*(tidx+stride)]) );
        }
        __syncthreads();
    }
    //thread zero writes to destination
    if(tidx == 0) res[bidx] = *acc;

    // if (tidx == 0) printf("block %02d end reduction \n", bidx);

}

template<typename T>
void multiplyWitness(T *res, T *witness){
    //wrapper for the kernel calling
    const size_t blockSize = 32; //number of threads 
    const int sharedMem = (blockSize+1) * (sizeof(T)+sizeof(uint64_t)); //dynamic shared mem, 64bit padding
    const size_t gridSizeMul = cuda_nCols;
    const size_t gridSizeAdd = cuda_nRows;
    cudaError_t err;

    //check if matrices and memory is allocated.
    if(!cuda_matrix_multiplication_ready) return; 


    // CPU_TIMER_INIT;
    
    // CPU_TIMER_START;
    multiplyWitnessKernel<fr_t><<<gridSizeMul, blockSize, sharedMem>>>(witness);
    CUDASYNC("Multiplication Kernel");
    sumWitnessKernel<fr_t><<<gridSizeAdd, blockSize, sharedMem>>>(res);
    CUDASYNC("Sum Kernel");
    // CPU_TIMER_END;
    // CPU_TIMER_PRINT("Gpu kernel multiplication ");


}

extern "C"
void multiplyWitnessCUDA(fr_t *res, fr_t *witness){
    multiplyWitness<fr_t>(res, witness);
}

extern "C"
void sparseMatrixLoadCUDA(const fr_t *A_data, const size_t *A_colidx, const size_t *A_indptr, const size_t A_NNZ, const size_t A_nRows, 
                          const fr_t *B_data, const size_t *B_colidx, const size_t *B_indptr, const size_t B_NNZ, const size_t B_nRows, 
                          const fr_t *C_data, const size_t *C_colidx, const size_t *C_indptr, const size_t C_NNZ, const size_t C_nRows, 
                          const size_t nCols_l){

     sparseMatrixLoad<fr_t, size_t>(A_data, A_colidx, A_indptr, A_NNZ, A_nRows, \
                                    B_data, B_colidx, B_indptr, B_NNZ, B_nRows, \
                                    C_data, C_colidx, C_indptr, C_NNZ, C_nRows, \
                                    nCols_l);

}

extern "C"
void freeManagedMatrix(){
    // frees all the allocated memory
    if (cuda_matrix_multiplication_ready == true){
        cudaFree(cuda_data);
        cudaFree(cuda_tmp_mul);
        cudaFree(MultiplicationDestinationPointer);
        cudaFree(cuda_colptr);
        cudaFree(cuda_rowidx);
        cudaFree(cuda_rowptr);
    }
    //todo
}


/////////////////////DEBUG STUFF//////////////////////
