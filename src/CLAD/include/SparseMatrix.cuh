// Sparse matrix implementation compatible with arecibo::r1cs::SparseMatrix<PrimeField>
#pragma once
#include <cassert>
#include <stdexcept>
#include <unistd.h>

#define CHECK(fmt) if (err != cudaSuccess){                                                                          \
                    printf("\n%s:%d " #fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));     \
                    throw std::bad_alloc();}


template<typename F>
class SparseMatrix_t {
    public:

    F      *data;        // All non-zero values in the matrix
    size_t *indices;     // Column indices
    size_t *indptr;      // Row information
    size_t nElements;    // Number of non-zero elements
    size_t cols;         // Number of columns
    size_t rows;         // Number of Rows

    void allocateUnifiedMemory(size_t nRow, size_t nCol, size_t nElements){
        cudaError_t err; 


        err = cudaMallocManaged(&data,    nElements * sizeof(F)); CHECK(data);
        err = cudaMallocManaged(&indices, nElements * sizeof(size_t)); CHECK(indices);
        err = cudaMallocManaged(&indptr, (nRow+1) * sizeof(size_t)); CHECK(indptr);
    }

    void deallocate(){
        //Dealocates managed memory on device
        cudaError_t err; 
        #define FREEANDCHECK(pointer) err = cudaFree(pointer); assert(err == cudaSuccess);

        FREEANDCHECK(data);
        FREEANDCHECK(indices);
        FREEANDCHECK(indptr);

        #undef FREEANDCHECK
    }

    // Constructor
    SparseMatrix_t(F *data_, size_t *indices_, size_t *indptr_, size_t nRows_, size_t nCols_, size_t nElements_)
        : nElements(nElements_), rows(nRows_), cols(nCols_){

        //Initializes matrix, copying from rust pointers
        allocateUnifiedMemory(nRows_, nCols_, nElements);

        //copy data
        memcpy(data, data_, nElements_*sizeof(F));
        memcpy(indices, indices_, nElements_*sizeof(size_t));
        memcpy(indptr, indptr_, (nCols_+1)*sizeof(size_t));
    }

    //Destructor
    ~SparseMatrix_t(){
        deallocate();
    }

    // For security, this matrix is readonly, writeback is done to a vector of F.   
    // void copyToRust(F *data, size_t *indices, size_t *indptr){
    //     //writes back to the rust pointers
    // }

    F get(size_t row, size_t col){
        //return element at [row][col]
        // Check if the given row and column indices are within bounds
        assert(row < rows && col < cols);

        // Find the range of column indices for the given row
        size_t start = indptr[row];
        size_t end = indptr[row + 1];

        // Search for the column index in the range
        for (int i = start; i < end; ++i) {
            if (indices[i] == col) {
                // If found, return the corresponding value
                return data[i];
            }
        }
        return 0;

    }

};


