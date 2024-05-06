#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <numeric> //for iota
#include <algorithm> //sor sort
#include <unordered_map>
#include <map>
#include <iostream>
#include <fr.cuh>

using std::vector; 
using std::unordered_map;

// Managared CUDA values

__managed__ fr_t   *data;        // All non-zero values in the matrix
__managed__ size_t *indices;     // Column indices
__managed__ size_t *multTarget;  // writing address for multiplication
__managed__ size_t *indptr;      // Row information
__managed__ size_t NZZ;          // Number of non-zero elements
__managed__ size_t nCols;        // Number of columns
__managed__ size_t nRows;        // Number of Rows in total
__managed__ size_t nRowsA;       // Number of Rows in A
__managed__ size_t nRowsB;       // Number of Rows in B
__managed__ size_t nRowsC;       // Number of Rows in C
__managed__ size_t *colptr;      // Column indices similar to indprt;
__managed__ size_t *rowDest;     // 

//Helper functions

vector<size_t> sort_by_frequency(const vector<size_t>& arr) {
    unordered_map<size_t, size_t> frequency;
    for (size_t num : arr) {
        frequency[num]++;
    }

    // Vector of indices to be sorted by frequency
    vector<size_t> indices(arr.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }

    // Sort indices by frequency of corresponding number in arr
    sort(indices.begin(), indices.end(), 
        [&arr, &frequency](size_t i, size_t j) {
            size_t num_i = arr[i], num_j = arr[j];
            // return frequency[num_i] > frequency[num_j] || 
            //       (frequency[num_i] == frequency[num_j] && num_i < num_j);

            if (frequency[num_i] == frequency[num_j])
                return num_i > num_j;
            else
                return frequency[num_i] > frequency[num_j];
        });

    return indices;
}

vector<size_t> invertPermutation(vector<size_t>&idx){
    //originally, you calculate the new vector as sorted[i] = unsorted[idx[i]]
    //by versing the permutation, it is sorted[inv_idx[i]] = unsorted[i];
    //this second form is better for cuda access patters, since we can match the read value to the thread idx.

    vector<size_t> inv_idx(idx.size());
    for(int i=0; i<idx.size(); i++){
        inv_idx[idx[i]] = i; 
    }
    return inv_idx;
}

//writes to permutationVector the permutation need to sort sortingKey in DESCENDING ORDER
template<typename T>
vector<size_t> sortKey(vector<T> sortingKey){
    vector<size_t> B(sortingKey.size());
    // Fill the vector with values from 0 to size-1
    iota(B.begin(), B.end(), 0);

    // Sort the indices vector based on values in B
    sort(B.begin(), B.end(), [&sortingKey](int i, int j) {return sortingKey[i] > sortingKey[j];});

    return B;
}

//Loads matrices and call preprocessing 
template<typename field>
void sparseMatrixLoad(field *A_data, size_t *A_indices, size_t *A_indptr, size_t A_NNZ, size_t A_nRows, 
                      field *B_data, size_t *B_indices, size_t *B_indptr, size_t B_NNZ, size_t B_nRows, 
                      field *C_data, size_t *C_indices, size_t *C_indptr, size_t C_NNZ, size_t C_nRows, 
                      size_t nCols_l){
    //allocate memory
    size_t ABC_nRows = A_nRows + B_nRows + C_nRows;
    size_t ABC_NNZ = A_NNZ+B_NNZ+C_NNZ;

    //Copy *indptr from A, reconstruct B, and C
    vector<size_t> ABC_indptr(ABC_nRows+1);
    size_t rowLen; 
    size_t index = 1;
    ABC_indptr[0] = 0;

    for(size_t i=0; i<A_nRows; i++){
        rowLen = A_indptr[i+1]-A_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
        index++;
    }
    for(size_t i=0; i<B_nRows; i++){
        rowLen = B_indptr[i+1]-B_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
        index++;
    }
    for(size_t i=0; i<C_nRows; i++){
        rowLen = C_indptr[i+1]-C_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
        index++;
    }

    //Calculate colWeight and rowWeight vectors
    vector<size_t> colWeight(nCols_l, 0); 
    vector<size_t> rowWeight(A_nRows + B_nRows + C_nRows, 0);

    for(size_t i=0; i<A_NNZ; i++){
        colWeight[A_indices[i]]++;
    }
    for(size_t i=0; i<B_NNZ; i++){
        colWeight[B_indices[i]]++;
    }
    for(size_t i=0; i<C_NNZ; i++){
        colWeight[C_indices[i]]++;
    }

    for(size_t i=0; i<ABC_nRows; i++){
        rowWeight[i] = ABC_indptr[i+1]-ABC_indptr[i];
    }


    //rowIndexer
    vector<size_t> rowIndexer(ABC_NNZ);
    for(size_t row=0; row<ABC_nRows; row++){
        for(size_t i = ABC_indptr[row]; i<ABC_indptr[row+1]; i++){
            rowIndexer[i]=row;
        }
    }
    auto posVector = sort_by_frequency(rowIndexer);

    //rowDestination
    vector<size_t> rowDestination(ABC_nRows);
    rowDestination[0] = rowIndexer[posVector[0]];
    for(int i=1, j=1; i<ABC_NNZ; i++){
        if (rowIndexer[posVector[i]] != rowDestination[j-1]){
            rowDestination[j] = rowIndexer[posVector[i]];
            j++;
        }
    }

    //generate new indptr for the weight ordered array - will be used by the multiplication to write the data to correct places.
    vector<size_t> ABC_indptr_sorted(ABC_nRows+1);
    ABC_indptr_sorted[0]=0;
    for(size_t i=0; i<ABC_nRows; i++){
        size_t target = rowDestination[i];
        size_t rowLen = ABC_indptr[target+1]-ABC_indptr[target];
        ABC_indptr_sorted[i+1] = ABC_indptr_sorted[i]+rowLen;
    }

    // At this point, we have 3 working arrays: ABC_data, ABC_Cols and posVector. By executing mul[i] = data[i] * witness[col[i]], 
    // the resultng mul vector will be ordered such that rows with the most ammount of elements will be first in memory.
    // (data and cols has not been copied over yet).
    // Next step is to sort these three arrays in such a way that the collums with more elements are first in memory. This
    // allows the multiplication kernel to keep witness[col[i]] in shared memory, and execute contiguous memory access. Best scenario is that every
    // warp is responsible for a colum, and loops over all the collums referent to this. 

    //Order by colweight, and use to load *data and populkate *indices
    int pointer = 0;
    vector<size_t> ABC_colIndex(A_NNZ+B_NNZ+C_NNZ);
    for(int i=0; i<A_NNZ; i++){
        ABC_colIndex[pointer] = A_indices[i];
        pointer++;
    }
    for(int i=0; i<B_NNZ; i++){
        ABC_colIndex[pointer] = B_indices[i];
        pointer++;
    }
    for(int i=0; i<C_NNZ; i++){
        ABC_colIndex[pointer] = C_indices[i];
        pointer++;
    }

    vector<size_t> columnSortKey = sort_by_frequency(ABC_colIndex);


    //Allocate managed memory on cuda.
    NZZ = A_NNZ + B_NNZ + C_NNZ;
    nCols = nCols_l;
    nRows = ABC_nRows;
    nRowsA = A_nRows;
    nRowsB = B_nRows;
    nRowsC = C_nRows;

    cudaError_t  err;
    err = cudaMallocManaged(&data, NZZ * sizeof(field));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&indices, NZZ * sizeof(size_t));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&multTarget, NZZ * sizeof(size_t));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&indptr, (nRows+1) * sizeof(size_t));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&colptr, (nCols+1) * sizeof(size_t));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&rowDest, (nRows) * sizeof(size_t));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    //copy data and indexes sorted;
    for(size_t i=0; i<ABC_NNZ; i++){
        size_t target = i;
        size_t source = columnSortKey[i];
        
        if (source < A_NNZ){ //source word located in A
            memcpy(&data[target], &A_data[source], sizeof(field));
            indices[target] = A_indices[source];
        }
        else if (source < A_NNZ+B_NNZ){ //source word located in B
            source = source - A_NNZ; 
            memcpy(&data[target], &B_data[source], sizeof(field));
            indices[target] = B_indices[source];
        }
        else{ //source word located in C
            source = source - A_NNZ - B_NNZ;  
            memcpy(&data[target], &C_data[source], sizeof(field));
            indices[target] = C_indices[source];
        }
    }

    //TODO: this can be optimized
    vector<size_t>RowsPermuted(NZZ);
    for(size_t i=0; i<NZZ; i++){
        RowsPermuted[i] = rowIndexer[columnSortKey[i]];
    }
    auto mTarget = sort_by_frequency(RowsPermuted);

    //This permutation is inverted, so one could address it as target[index[i]] = source[i]
    //this is usefull because the source can be indexed by threadId, making so a whole warp writes the scather.
    for(size_t i=0; i<NZZ; i++){
        // multTarget[i] = mTarget[i]; //not inverted
        multTarget[mTarget[i]] = i; //not inverted
    }

    //copy rowDest
    for(int i=0; i<nRows; i++)
        rowDest[i] = rowDestination[i];

    //new indptr
    for(int i=0; i<NZZ+1; i++)
        indptr[i] = ABC_indptr_sorted[i];

    // Generate Colptr for helping the multiplication
    colptr[0] = 0; 
    size_t count=1; 
    size_t curr=indices[0]; 
    for(size_t i = 1; i<NZZ; i++){
        if (indices[i]==curr)
            continue;
        else{
            colptr[count] = i;
            curr = indices[i];
            count++;
        }
    }
    colptr[count+1] = NZZ; //last element is NNZ by definition


    //Free any allocated temporaries
}

template<typename field>
__global__  void multiplyWitnessKernel(field *ABCZ, field *witness){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    unsigned gdim = gridDim.x; 
    __shared__ field m; 

    // We are going to use a warp for multiplication -> 32 threads, and fill the SM with 2048 threads (64 warps, one block per warp)
    
    for(int i = bidx; i<nCols; i+=gdim){ //loop over the columns
        m = witness[indices[i]];  //load multiplicand to shared memory

        size_t colStart = colptr[i + bidx]; //points at start of column of each block
        size_t colEnd = colptr[i+1 + bidx]; //points at end of column of each block

        for (size_t j=colStart; j<colEnd; j+=bdim){

            if((j+tidx) >= colEnd )break; //if lenght of col is not a multiple of 32

            field res = mul(m, data[j+tidx]); 

            ABCZ[multTarget[j+tidx]] = res; //multiply and add to correct destination //TODO: Change for Field after int debug
        }
    }
}


template<typename field>
__global__  void sumWitnessKernel(field *res, field *ABCZ){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    unsigned gdim = gridDim.x; 

    extern __shared__ field accumulator[]; //make sure this array is of lenght bdim
    
    // We are going to use a warp for sum -> 32 threads, and use 1warp = 1block to run
    for(int i = bidx; i<nRows; i+=gdim){
        size_t start = indptr[i];
        size_t end = indptr[i +1];
        field m = zero(); //fr.zero()

        //group of threads sum to shared mem
        size_t maxIndex = end-((end-start)%bdim);
        for (size_t j = start; j<maxIndex; j+=bdim){
            m = sum(m, ABCZ[j+tidx]);
        }
        //last bits
        if(tidx < (end-start)%bdim){
            m = sum(m, ABCZ[maxIndex+tidx]);
        }

        //reduce shared memory
        accumulator[tidx]=m;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride>>=1){
            if (tidx < stride){
                accumulator[tidx] = sum(accumulator[tidx], accumulator[tidx+stride]);
            }
            __syncthreads();
        }

        //thread zero writes to result
        if(tidx == 0) res[rowDest[i]] = accumulator[0];
    }
    __syncthreads();
}

template<typename field>
vector<field> multiplyWitnessCPU(field *witness){
    //runs the alg using CPU, usefull for debugging the preprocessing step and kernels

    //multiply
    vector<field> m (NZZ);
    for(size_t i=0; i<NZZ; i++){
        // m[i] = data[multTarget[i]]*witness[indices[multTarget[i]]]; //Not inverted Multarget
        m[multTarget[i]] = mul(data[i], witness[indices[i]]); //invertedMulTarget
    }

    //add
    vector<field> resultUnordered(nRows, 0);
    for(size_t i=0; i<nRows; i++){
        size_t start = indptr[i];
        size_t end   = indptr[i+1];
        for(size_t j=start; j < end; j++){
            resultUnordered[i] =sum(resultUnordered[i],  m[j]);
        }
    }

    //reorder
    vector<field> res(nRows);

    for(int i=0; i<nRows; i++)
        res[rowDest[i]] = resultUnordered[i];

    return res;
}

__managed__ fr_t *ABCZ, *res_cuda;
__managed__ fr_t *w_cuda;

extern "C"
void multiplyWitnessCUDA(fr_t *res, fr_t *witness){
    #ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        }                                                                                                     
    #endif

    int blockSize = 32;
    int gridSize = 64;
    int sharedMem = blockSize * sizeof(fr_t);
    cudaError_t err;

    cudaMallocManaged(&ABCZ, sizeof(fr_t)* NZZ);
    cudaMallocManaged(&res_cuda, sizeof(fr_t)* nRows);
    cudaMallocManaged(&w_cuda, sizeof(fr_t)* nCols);

    memcpy(w_cuda, witness, nCols*sizeof(fr_t));

    multiplyWitnessKernel<<<gridSize, blockSize>>>(ABCZ, w_cuda); 
    CUDASYNC("Multiply");

    sumWitnessKernel<<<gridSize, blockSize, sharedMem>>>(res_cuda, ABCZ);
    CUDASYNC("Sum");

    memcpy(res, res_cuda, nRows*sizeof(fr_t));

}



extern "C"
void freeManagedMatrix(){
    // frees all the allocated memory

    cudaFree(data);
    cudaFree(indices);
    cudaFree(multTarget);
    cudaFree(indptr);
    cudaFree(colptr);
    cudaFree(rowDest);
}


/////////////////////DEBUG STUFF//////////////////////
