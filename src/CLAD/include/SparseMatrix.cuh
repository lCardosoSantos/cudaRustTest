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

    int64_t *idx;          // used on the preprocessing phase.

#ifdef ASYNCLOAD
    cudaEvent_t loaded;  // for Asynchronous load
    cudaStream_t loadingStream; //
#else
    bool loaded = false; 
#endif

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

        free(idx);
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

        idx = (int64_t *)malloc(nElements_ * sizeof(int64_t));
        for(int i=0; i<nElements_; i++) idx[i]=i;

        preprocess();
        loaded = true;
    }

    // Empty constructor
    SparseMatrix_t(const SparseMatrix_t &) = default;

    //Destructor
    ~SparseMatrix_t(){
        deallocate();
    }

    //reeorganize matrix for fast memory access on cuda
    size_t qpart(size_t *array, size_t low, size_t high){
        size_t i = low-1;

        size_t pivot = array[high];
        for(size_t j=low;j<high;j++)
        {
            if(array[j]<=pivot)
            {
                i++;
                // swap(nums[i],nums[j]);
                std::swap(data[i], data[j]);
                std::swap(indices[i], indices[j]);
                std::swap(idx[i], idx[j]);
            }
        }
        // swap(nums[i+1],nums[high]);
        std::swap(data[i+1], data[high]);
        std::swap(indices[i+1], indices[high]);
        std::swap(idx[i+1], idx[high]);
        return i+1;

    }
    void cqsort(size_t *array, size_t low, size_t high){
        if(low<high){
            size_t pivot = qpart(array, low, high);
            cqsort(array, low, pivot-1);
            cqsort(array, pivot+1, high);
        }
    }

    void preprocess(){
        cqsort(indices, 0, nElements-1);
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


