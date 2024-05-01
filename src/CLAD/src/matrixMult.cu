#include <stdio.h>
#include <assert.h>
#include "fr.cuh"
#include "SparseMatrix.cuh"
#include "matrixMult.cuh"
#include <vector>

using namespace std;


SparseMatrix_t<fr_t> *A, *B, *C; 

extern "C"
void matrix_load(void *data, size_t *indices, size_t *indptr, 
                size_t nElements, size_t cols, size_t rows, int matrixIndex){

    SparseMatrix_t tmp = SparseMatrix_t<fr_t>((fr_t*)data, indices, indptr,
                            nElements, cols, rows);

    //TODO: FFI with enums                                      
    switch (matrixIndex){
        case 0:
            A = &tmp;
            break;
        case 1:
            B = &tmp;
            break;
        case 2:
            C = &tmp;
            break;
    }
                
}

//Load implies also any preprocessing, if Any
bool checkLoad(){
    #ifdef ASYNCLOAD    
    //check all matrices are loaded, if not, lock until finished
    //TODO: Async load check
    #else
    return (A->loaded && B->loaded && C->loaded);

    #endif 
}


template<typename field>
__global__ void reorderWithIndex(field *array, size_t *index, size_t len){
    int64_t cycles [len];
    for(int i=0; i<len; i++) cycles[i] = index[i]; //the index is destroyed during reordering
    field value; 

    for(int i=0; i<len; i++){
        if (cycles[i] == -1) continue; //already processed
        copy(value, array[i]);
        int x = i;
        int y = cycles[i];

        while(y!=-1){
            cycles[i] = -1; //mark as processed
            copy(array[x], array[y]);
            x=y;
            y=cycles[x];
        }
        copy(array[x], value);
        cycles[x]=-1;
    }
    
}

template<typename field>
__global__ void multiplyWitness_CPU(field *AZ, field *BZ, field *CZ, field *witness){
    //allocate tmp
    field *Af = malloc(A->nElements*sizeof(field)); 
    field *Bf = malloc(B->nElements*sizeof(field)); 
    field *Cf = malloc(C->nElements*sizeof(field));

    //multiplications
    //writing back to sorted array
    for(size_t i=0; i<A->nElements; i++)
        mul(&Af[A->idx[i]], &A->data[i], &witness[A->indices[i]] );
    for(size_t i=0; i<B->nElements; i++)
        mul(&Bf[B->idx[i]], &B->data[i], &witness[B->indices[i]] );
    for(size_t i=0; i<C->nElements; i++)
        mul(&Cf[C->idx[i]], &C->data[i], &witness[C->indices[i]] );

    //zero destination
    for(int i=0; i < A->rows; i++){
        AZ[i] = field();
        BZ[i] = field();
        CZ[i] = field();
    }

    //sum
    for(size_t row=0; row< A->rows; row++){
        size_t start = A->indptr[row]; 
        size_t end   = A->indptr[row+1];
        for(int i=start; i<end; i++)
            sum(&AZ[row], &AZ[row], &AZ[i]);
    }
    for(size_t row=0; row< B->rows; row++){
        size_t start = B->indptr[row]; 
        size_t end   = B->indptr[row+1];
        for(int i=start; i<end; i++)
            sum(&BZ[row], &BZ[row], &BZ[i]);
    }
    for(size_t row=0; row< C->rows; row++){
        size_t start = C->indptr[row]; 
        size_t end   = C->indptr[row+1];
        for(int i=start; i<end; i++)
            sum(&CZ[row], &CZ[row], &CZ[i]);
    }

    //free resources
    free(Af);
    free(Bf);
    free(Cf);
}

/***
 * Calculates the cross term.
 * The memory management is done on the rust size, assuming that information was gathered by the matrix load function.
*/
extern "C"
void genWitness(void *AZ,
                  void *BZ,
                  void *CZ,
                  void *witness){


    
    if (checkLoad() == false){
        printf("FATAL: Matrices are not loaded");
        exit(-1);
    }

    // Call kernel multiplyWitness
    cudaError_t err;

    // multiplyWitness<fp_t><<<NTHREADS, NBLOCKS>>>((fp_t*)AZ, (fp_t*)BZ, (fp_t*)CZ, 
    //                                              (fp_t*)witness);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){                         
        printf("\n%s:%d  Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    }

    return;
}

///////////////////// Second method ///////////////////
// Managared CUDA values


__managed__ fr_t   *data;        // All non-zero values in the matrix
__managed__ size_t *indices;     // Column indices
__managed__ size_t *multTarget;  // writing address for multiplication
__managed__ size_t *indptr;      // Row information
__managed__ size_t NZZ;    // Number of non-zero elements
__managed__ size_t nCols;        // Number of columns
__managed__ size_t nRows;        // Number of Rows in total
__managed__ size_t nRowsA;       // Number of Rows in A
__managed__ size_t nRowsB;       // Number of Rows in B
__managed__ size_t nRowsC;       // Number of Rows in C
__managed__ size_t *colptr;      // Column indices similar to indprt;


//Helper functions

//writes to permutationVector the permutation need to sort sortingKey in DESCENDING ORDER
template<typename T>
vector<size_t> sortKey(T *sortingKey, size_t len){
    vector<T> B = (sortingKey, sortingKey+len);
    // Create a vector to store the indices
    vector<int> indices(B.size());
    // Fill the vector with values from 0 to size-1
    iota(indices.begin(), indices.end(), 0);

    // Sort the indices vector based on values in B
    sort(indices.begin(), indices.end(), [&B](int i, int j) {
        return B[i] > B[j];
    });

    return indices;
}

//writes to ouput the input on the positions of Permutation vector
template<typename T>
void applyPermutation(T *output, T *input, vector<size_t> permutationVector, size_t len){
    for(size_t i=0; i<len; i++){
        memcpy(&output[permutationVector[i]], &input[i], sizeof(T));
    }
}

//Reverses a permutaion applied with applyPermitation
template<typename T>
void undoPermutation(T *output, T *input, vector<size_t> permutationVector, size_t len){
    for(size_t i=0; i<len; i++){
        memcpy(&output[i], &input[permutationVector[i]], sizeof(T));
    }
}



//Loads matrices and call preprocessing 
template<typename field>
void sparseMatrixLoad(field *A_data, size_t *A_indices, size_t *A_indptr, size_t A_NNZ, size_t A_nRows, 
                      field *B_data, size_t *B_indices, size_t *B_indptr, size_t B_NNZ, size_t B_nRows, 
                      field *C_data, size_t *C_indices, size_t *C_indptr, size_t C_NNZ, size_t C_nRows, 
                      size_t nCols_l){
    //allocate memory
    size_t ABC_nRows = A_nRows + B_nRows + C_nRows;

    //Copy *indptr from A, reconstruct B, and C
    vector<size_t> ABC_indptr(ABC_nRows+1);
    size_t rowLen; 
    size_t index = 1;
    ABC_indptr[0] = [0];

    for(size_t i=0; i<A_nRows; i++){
        rowLen = A_indptr[i+1]-A_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
    }
    for(size_t i=0; i<B_nRows; i++){
        rowLen = B_indptr[i+1]-B_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
    }
    for(size_t i=0; i<C_nRows; i++){
        rowLen = C_indptr[i+1]-C_indptr[i];
        ABC_indptr[index] = ABC_indptr[index-1]+rowLen;
    }

    //Calculate colWeight and rowWeight vectors
    vector<size_t> colWeight(nCols_l); iota(colWeight.begin(), colWeight.end(), 0);
    vector<size_t> rowWeight(A_nRows + B_nRows + C_nRows); iota(rowWeight.begin(), rowWeight.end(), 0);

    for(size_t i=0; i<nCols_l; i++){
        colWeight[ A_indices[i] ]++;
        colWeight[ B_indices[i] ]++;
        colWeight[ C_indices[i] ]++;
    }

    for(size_t i=0; i<ABC_nRows; i++){
        rowWeight[i] = ABC_indptr[i+1]-ABC_indptr[i];
    }

    //generate rowDestination array
    vector<size_t> rowDestination = sortKey(rowWeight, rowWeight.size);

    //generate posVector
    //posVector is a 1 to 1 array that points to where a value should be written to after a multiplication
    vector<size_t> posVector(ABC_nRows);
    size_t indexer = 0
    for(size_t row=0; row<ABC_nRows; row++){
        size_t source = rowDestination[i];
        size_t start  = ABC_indptr[source];
        size_t end    = ABC_indptr[source+1];

        for(size_t i=start; i<end; i++){
            posVector[indexer] = i; 
        }

    }
    //generate new indptr for the weight ordered array - will be used by the multiplication to write the data to correct places.
    vector<size_t> ABC_indptr_sorted(ABC_nRows+1);
    ABC_indptr_sorted [0] = 0;
    size_t run = 0;
    for (int i=0, i<ABC_indptr_sorted; i++){
        size_t sourceRow = rowDestination[i]
        size_t rowWeight = ABC_indptr[sourceRow+1] - ABC_indptr[sourceRow];
        run += rowWeight;
        ABC_indptr_sorted[i+1] = run; 
    }

    // At this point, we have 3 working arrays: ABC_data, ABC_Cols and posVector. By executing mul[i] = data[i] * witness[col[i]], 
    // the resultng mul vector will be ordered such that rows with the most ammount of elements will be first in memory.
    // (data and cols has not been copied over yet).
    // Next step is to sort these three arrays in such a way that the collums with more elements are first in memory. This
    // allows the multiplication kernel to keep witness[col[i]] in shared memory, and execute contiguous memory access. Best scenario is that every
    // warp is responsible for a colum, and loops over all the collums referent to this. 
    // TODO: Is a vector pointing the limits of each column useful here?


    //Order by colweight, and use to load *data and populkate *indices
    int pointer = 0;
    vector<size_t> ABC_colIndexUnsorted(A_NNZ+B_NNZ+C_NNZ);
    for(int i=0; i<A_NNZ; i++){
        ABC_colIndex[pointer] = A_indices[i];
    }
    for(int i=0; i<B_NNZ; i++){
        ABC_colIndex[pointer] = B_indices[i];
    }
    for(int i=0; i<C_NNZ; i++){
        ABC_colIndex[pointer] = C_indices[i];
    }

    vector<size_t> columnSortKey = sortKey(ABC_colIndex, ABC_colIndex.size() )


    //Allocate managed memory on cuda.
    nElements = A_NNZ + B_NNZ + C_NNZ;
    nCols_l = nCols_l;
    nRows = ABC_nRows
    nRowsA = A_nRows;
    nRowsB = B_nRows;
    nRowsC = C_nRows;

    cudaError_t  err;
    err = cudaMallocManaged(&data, nElements * sizeof(field));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&indices, nElements * sizeof(field));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&indptr, (ABC_nRows+1) * sizeof(field));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    err = cudaMallocManaged(&colptr, (ABC_nCols+1) * sizeof(field));
    if (err != cudaSuccess){
        printf("FATAL: Cuda memory allocation failed!\n");
        printf("Error: %d: %s\n", err, cudaGetErrorName(err));
        exit(0xf1);
    }

    //copy data sorted
    for(size_t i=0; i<A_NNZ; i++){
        memcpy(&data[columnSortKey[i]], &A_data[i], sizeof(field));
        indices[columnSortKey[i]] = A_indices[i];
    }
    for(size_t i=0; i<B_NNZ; i++){
        memcpy(&data[columnSortKey[A_NNZ+i]], &B_data[i], sizeof(field));
        indices[columnSortKey[A_NNZ+i]] = A_indices[i];
    }
    for(size_t i=0; i<A_NNZ; i++){
        memcpy(&data[columnSortKey[A_NNZ+B_NNZ+i]], &C_data[i], sizeof(field));
        indices[columnSortKey[A_NNZ+B_NNZ+i]] = A_indices[i];
    }

    for(size_t i=0; i<nElements; i++)
        multTarget[columnSortKey[i]] = posVector[i];

    // Generate Colptr for helping the multiplication
    colptr[0] = 0; 
    size_t count=1; 
    size_t curr=indices[0]; 
    for(size_t i = 1; i<NZZ; i++){
        if (indices[i]==curr)
            continue;
        else
            colptr[count] = i;
            curr = indices[i];
            count++;
    }


    //Free any allocated temporaries
}

template<typename field>
__global__  void multiplyWitnessKernel(field *ABCZ, field *witness){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    unsigned gdim = gridDim.x; 

    __shared__ uint32_t w[(4*8+1)*8];

    // We are going to use a warp for multiplication -> 32 threads, and fill the SM with 2048 threads (64 warps, one block per warp)
    
    for(int i = 0; i<nCols; i+=gdim){ //loop over the columns
        m = witness[indices[j+tid]]; //load multiplicand to shared memory

        size_t colStart = colptr[i + bidx]; //points at start of column of each block
        size_t colEnd = colptr[i+1 + bidx]; //points at end of column of each block

        for (size_t j=colStart; j<colEnd; j+=bdim){
            //Load multipliers to shared memory
            //load from J to J+32
            uint32_t *input = &(data[j]);
            for(int i=0; i<8; i++){
                w[33*i+4*tid] = input[32*i+4*tid];
            }
            field res = m * w[tidx*4 + tidx%4];

            ABCZ[multTarget[j+tid]] = res; //multiply and add to correct destination //TODO: Change for Field after int debug
        }
    }
}

template<typename field>
__global__  void sumWitnessKernel(field *resUnsorted, field *ABCZ){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
     unsigned gdim = gridDim.x; 

    __shared__ field m[bdim];
    // We are going to use a warp for sum -> 32 threads, and use 1warp = 1block to run
    for(int i = 0; i<nRows; i+=gdim){
        size_t start = indptr[i + bidx];
        size_t start = indptr[i +1 + bidx];

        //group of threads sum to shared mem
        for (size_t j = start; j<end; j+=bdim){
            m[tidx] =+ ABCZ[j+tidx];
        }

        //reduce shared memory
        for (int s = bdim.x / 2; s > 0; s>>=1){
            if (tidx < s){
                m[tid] += m[tid+s];
            }
            __syncthreads();
        }

        //thread zero writes to result
        if(tidx == 0) resUnsorted[i+bidx] = m[0];
    }

}

template<typename field>
void writeSorted(field *rustPointerA, field *rustPointerB, field *rustPointerC, field *resUnsorted){
    //uses colum sorting information to write data back into the correct pointers.
}

extern "C"
void freeManagedMatrix(){
    // frees all the allocated memory

}


/////////////////////DEBUG STUFF//////////////////////

extern "C" void easy(void *data){
    
    printf(">>> %p\n", data);
    // fp_t scalar;
    // uint64_t *p = (uint64_t *)data;

    // scalar.from_mont(p[0], p[1], p[2], p[3]);
    
    
    printf(">>>raw ");
    for (int i =0; i<4; i++)
        printf("%016x ", ((uint64_t*)data)[i]); 

    // field_printh("from mont", scalar);


}   
