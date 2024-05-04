//INTERNAL EXPERIMENTATION ONLY
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <numeric> //for iota
#include <algorithm> //sor sort
#include <unordered_map>
#include <map>
#include <iostream>

using std::vector; 
using std::unordered_map;

class fr_t {
    public:
    uint64_t _[4];
    fr_t(const fr_t &) = default;
    ~fr_t() = default;
    
    // Default constructor
    fr_t() : _{0, 0, 0, 0} {}  // Initialize all elements to zero

    fr_t(int a){
        _[0] = a;
        _[1] = 0;
        _[2] = 0;
        _[3] = 0;

    }

    bool operator>(const fr_t& y) const{
        return _[0] > y._[0];
    }

    bool operator<(const fr_t& y) const{
        return _[0] < y._[0];
    }

    // Multiplication operator
    fr_t operator*(const fr_t& y) const {
        return fr_t(static_cast<int>(_[0] * y._[0])); 
    }

    // Addition assignment operator
    fr_t& operator+=(const fr_t& y) {
        _[0] += y._[0]; 
        return *this;
    }


};



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
        ABC_indptr[index] = A_indptr[index-1]+rowLen;
        index++;
    }
    for(size_t i=0; i<B_nRows; i++){
        rowLen = B_indptr[i+1]-B_indptr[i];
        ABC_indptr[index] = B_indptr[index-1]+rowLen;
        index++;
    }
    for(size_t i=0; i<C_nRows; i++){
        rowLen = C_indptr[i+1]-C_indptr[i];
        ABC_indptr[index] = C_indptr[index-1]+rowLen;
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

                // //generate rowDestination array
                // vector<size_t> rowDestination = sort_by_frequency(rowWeight);

                // //generate posVector
                // //posVector is a 1 to 1 array that points to where a value should be written to after a multiplication
                // vector<size_t> posVector(ABC_NNZ);
                // size_t indexer = 0;
                // for(size_t row=0; row<ABC_nRows; row++){
                //     size_t source = rowDestination[row];
                //     size_t start  = ABC_indptr[source];
                //     size_t end    = ABC_indptr[source+1];

                //     for(size_t i=start; i<end; i++){
                //         posVector[indexer] = i; 
                //         indexer++;
                //     }

                // }

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

    for(size_t i=0; i<NZZ; i++)
        multTarget[i] = mTarget[i];

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

    __shared__ uint32_t w[(4*8+1)*8];
    field m; 

    // We are going to use a warp for multiplication -> 32 threads, and fill the SM with 2048 threads (64 warps, one block per warp)
    
    for(int i = 0; i<(nCols+gdim); i+=gdim){ //loop over the columns
        if((i+bidx)>= nCols) break;
        m = witness[indices[i+tidx]];  //load multiplicand to shared memory

        size_t colStart = colptr[i + bidx]; //points at start of column of each block
        size_t colEnd = colptr[i+1 + bidx]; //points at end of column of each block

        for (size_t j=colStart; j<(colEnd+bdim); j+=bdim){
            //Load multipliers to shared memory
            //load from J to J+31
            uint32_t *input = (uint32_t*)(&(data[j]));
            for(int i=0; i<8; i++){
                w[33*i+4*tidx] = input[32*i+4*tidx];
            }

            if((j+tidx) >= colEnd )break; //if lenght of col is not a multiple of 32

            field res = m * w[tidx*4 + tidx%4];

            ABCZ[multTarget[j+tidx]] = res; //multiply and add to correct destination //TODO: Change for Field after int debug
        }
    }
}

template<typename field>
__global__  void sumWitnessKernel(field *resUnsorted, field *ABCZ){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
     unsigned gdim = gridDim.x; 

    field m[bdim];
    m[tidx] = 0; //fr.zero()

    // We are going to use a warp for sum -> 32 threads, and use 1warp = 1block to run
    for(int i = 0; i<nRows; i+=gdim){
        size_t start = indptr[i + bidx];
        size_t end = indptr[i +1 + bidx];

        //group of threads sum to shared mem
        for (size_t j = start; j<end; j+=bdim){
            m[tidx] =+ ABCZ[j+tidx];
        }
        //extra
        int j=(end/bdim)*bdim;
        if (j+tidx < end) m[tidx]+=ABCZ[j+tidx];

        //reduce shared memory
        for (int s = bdim / 2; s > 0; s>>=1){
            if (tidx < s){
                m[tidx] += m[tidx+s];
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

template<typename field>
vector<field> multiplyWitnessCPU(field *witness){
    //runs the alg using CPU, usefull for debugging the preprocessing step and kernels

    //multiply
    vector<field> m (NZZ);
    for(size_t i=0; i<NZZ; i++)
        m[i] = data[multTarget[i]]*witness[indices[multTarget[i]]];

    //add
    vector<field> resultUnordered(nRows, 0);
    for(size_t i=0; i<nRows; i++){
        size_t start = indptr[i];
        size_t end   = indptr[i+1];
        for(size_t j=start; j < end; j++){
            resultUnordered[i] += m[j];
        }
    }

    //reorder
    vector<field> res(nRows);

    for(int i=0; i<nRows; i++)
        res[rowDest[i]] = resultUnordered[i];

    return res;
}


// int main() {
//     std::vector<size_t> nums = {1, 6, 8, 0, 2, 5, 9, 1, 2, 6, 7, 8, 9, 0, 1, 8, 3, 4, 7, 1, 8, 6, 6};
//     // std::vector<size_t> nums = {0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4};
//     std::vector<size_t> sortedIndices = sort_by_frequency(nums);

//     std::cout << "Sorted indices by frequency and value: ";
//     for (int index : sortedIndices) {
//         std::cout << index << ", ";
//     }
//     std::cout << std::endl;

//     printf("applied: ");
//     for(int i=0; i<nums.size(); i++)
//         printf("%d, ", nums[sortedIndices[i]]);

//     std::cout << std::endl;

//     return 0;
// }

void pfr(fr_t *a, size_t len){
    for(int i=0; i<len; i++)
        printf("%lu, ", a[i]._[0]);
    printf("\n");
}

void pfr(vector<fr_t> &a){
    for(int i=0; i<a.size(); i++)
        printf("%lu, ", a[i]._[0]);
    printf("\n");
}

int main(){

    printf("test using int as base for the multiplication\n");
    
    //vars
    size_t nCols_l = 20;
    size_t A_nRows = 10;
    size_t B_nRows = 0;
    size_t C_nRows = 0;

    fr_t      A_data[25] = {133, 229, 7, 19, 96, 45, 113, 170, 32, 75, 71, 123, 207, 55, 51, 37, 197, 197, 133, 244, 170, 200, 163, 161, 48};
    size_t A_indices[25] = {17, 2, 12, 17, 1, 4, 4, 10, 18, 3, 4, 5, 15, 17, 13, 15, 18, 1, 2, 16, 6, 8, 15, 6, 8};
    size_t A_indptr[11] = {0, 1, 4, 6, 9, 14, 17, 18, 20, 23, 25};
    size_t A_NNZ = 25;

    fr_t w[20] = {37, 59, 0, 138, 51, 0, 11, 0, 0, 0, 87, 46, 61, 0, 140, 59, 0, 0, 214, 0}; //{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    
    sparseMatrixLoad<fr_t>(A_data, A_indices, A_indptr, A_NNZ, A_nRows,
                           NULL, NULL, NULL, 0, 0,
                           NULL, NULL, NULL, 0, 0,
                           nCols_l);



    auto res = multiplyWitnessCPU<fr_t>(w); //[255, 343, 538, 574, 533]
    pfr(res);

    printf("end\n");

}

