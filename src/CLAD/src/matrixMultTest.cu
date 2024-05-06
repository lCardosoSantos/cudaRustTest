//INTERNAL EXPERIMENTATION ONLY
//Does matrix Multiplication using a dummy fr_t
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include <vector>
#include <numeric> //for iota
#include <algorithm> //sor sort
#include <unordered_map>
#include <map>
#include <iostream>
#include <chrono>

using std::vector; 
using std::unordered_map;
using namespace std::chrono;


    #ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        }                                                                                                     
    #endif


class fr_t {
    public:
    __managed__ uint64_t _[4];
    fr_t(const fr_t &) = default;
    ~fr_t() = default;
    
    // Default constructor
    __host__ __device__ fr_t() : _{0, 0, 0, 0} {}  // Initialize all elements to zero

    __host__ __device__ fr_t(int a){
        _[0] = a;
        _[1] = a;
        _[2] = a;
        _[3] = a;

    }

    __host__ __device__ bool operator>(const fr_t& y) const{
        return _[0] > y._[0];
    }

    __host__ __device__ bool operator==(const fr_t& y) const{
        return _[0] == y._[0];
    }

    __host__ __device__ bool operator<(const fr_t& y) const{
        return _[0] < y._[0];
    }

    // Multiplication operator
    __host__ __device__ fr_t operator*(const fr_t& y) const {
        return fr_t(static_cast<int>(_[0] * y._[0])); 
    }

    // Addition assignment operator
    __host__ __device__ fr_t& operator+=(const fr_t& y) {
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


    //TODO: BUG HERE ON INDPTR CREATION
    vector<size_t> posVector = sort_by_frequency(rowWeight); //how to sort the array such that rows are in order.

    vector<size_t> rowDestination(ABC_nRows);
    for(int i=0; i<ABC_nRows; i++){
        rowDestination[i] = posVector[i];
    }

    //generate new indptr for the weight ordered array - will be used by the multiplication to write the data to correct places.
    vector<size_t> ABC_indptr_sorted(ABC_nRows+1);
    ABC_indptr_sorted[0]=0;
    for(size_t i=0; i<ABC_nRows; i++){
        size_t source = rowDestination[i];
        size_t rowLen = ABC_indptr[source+1]-ABC_indptr[source];
        ABC_indptr_sorted[i+1] = ABC_indptr_sorted[i]+rowLen;
    }

    //for composint the permutation
    vector<size_t> rowIndexer(ABC_NNZ);
    for(int row = 0; row<ABC_nRows; row++){
        for(size_t i = ABC_indptr[row]; i<ABC_indptr[row+1]; i++){
            rowIndexer[i]=row;
        }
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
    for(int i=0; i<nRows+1; i++)
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
__global__  void multiplyWitnessKernel_old(field *ABCZ, field *witness){
    unsigned tidx = threadIdx.x;
    unsigned bidx = blockIdx.x;
    unsigned bdim = blockDim.x;
    unsigned gdim = gridDim.x; 

    __shared__ uint32_t w[9*32]; // field=8 words + 1 for bank align, time 32 threads in a block.
    field m; 

    // We are going to use a warp for multiplication -> 32 threads, and fill the SM with 2048 threads (64 warps, one block per warp)
    
    for(int i = bidx; i<nCols; i+=gdim){ //loop over the columns
        m = witness[indices[i]];  //load multiplicand to shared memory

        size_t colStart = colptr[i + bidx]; //points at start of column of each block
        size_t colEnd = colptr[i+1 + bidx]; //points at end of column of each block

        for (size_t j=colStart; j<colEnd; j+=bdim){
            //Load multipliers to shared memory
            //load from J to J+31
            // uint32_t *baseIndex = (uint32_t*)(&(data[j+tidx]));
            size_t dataIndex = j+tidx;
            uint32_t *wPointer = (uint32_t *)(&data[dataIndex]);


            for(int ii=0; ii<8; ii++){
                int sharedIndex = tidx * 9 + ii;
                w[sharedIndex] = wPointer[ii];
            }

            if((j+tidx) >= colEnd )break; //if lenght of col is not a multiple of 32

            field *operand = reinterpret_cast<field*>(&w[tidx*9]);
            field res = m * (*operand);

            ABCZ[multTarget[j+tidx]] = res; //multiply and add to correct destination //TODO: Change for Field after int debug
        }
    }
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

            field res = m * data[j+tidx]; 

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
        field m = 0; //fr.zero()

        //group of threads sum to shared mem
        size_t maxIndex = end-((end-start)%bdim);
        for (size_t j = start; j<maxIndex; j+=bdim){
            m += ABCZ[j+tidx];
        }
        //last bits
        if(tidx < (end-start)%bdim){
            m += ABCZ[maxIndex+tidx];
        }

        //reduce shared memory
        accumulator[tidx]=m;
        __syncthreads();
        for (int stride = bdim / 2; stride > 0; stride>>=1){
            if (tidx < stride){
                accumulator[tidx] += accumulator[tidx+stride];
            }
            __syncthreads();
        }

        //thread zero writes to result
        if(tidx == 0) res[rowDest[i]] = accumulator[0];
        // if(tidx == 0) res[i] = accumulator[0];
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
        m[multTarget[i]] = data[i]*witness[indices[i]]; //invertedMulTarget
    }

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

template<typename field>
vector<field> multiplyWitnessCPU_straight(field *witness_s, field *data_s, size_t *indices_s, size_t *indptr_, 
                                         size_t nCols, size_t nRows_, size_t NZZ){
    //runs the alg using CPU, usefull for debugging the preprocessing step and kernels
    
    vector<field> resCPU(nRows_, 0);

    for(size_t i=0; i<nRows_; i++){
        size_t start = indptr_[i];
        size_t end   = indptr_[i+1];
        for(size_t j=start; j < end; j++){
            field m = data_s[j]*witness_s[indices_s[j]];
            resCPU[i] += m;
        }
    }
    return resCPU;
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

// __managed__ fr_t ABCZ[A_NNZ*3], res[A_nRows*3];
    __managed__ fr_t *ABCZ, *res_cuda;
    __managed__ fr_t *w_cuda;

int main(){
    cudaError_t err;
    size_t cols   = 79998;//4;//7999846;
    size_t nRows_ = 98250;//4;//9825045;
    size_t nZ     = 1046754;//6; //104675466; 
    // size_t data_limit = 10; //for easier debugging


    fr_t   *data_    = (fr_t *)malloc(nZ * sizeof(fr_t));
    fr_t   *wit      = (fr_t *)malloc(cols * sizeof(fr_t));
    size_t *indices_ = (size_t *)malloc(nZ * sizeof(size_t));
    size_t *indptr_  = (size_t *)malloc((nRows_ + 1) * sizeof(size_t));


    //random matrix
    // Fill row_ptrs with 0 initially
    for (size_t i = 0; i < nRows_+1; ++i) {
        indptr_[i] = 0;
    }

    // Randomly distribute nonzeros across the rows
    for (size_t i = 0; i < nZ; ++i) {
        size_t row = rand() % nRows_;
        indptr_[row + 1]++;
    }

    // Convert the count into actual indices_
    for (size_t i = 1; i <= nRows_; ++i) {
        indptr_[i] += indptr_[i - 1];
    }

    // Assign random columns and values to each entry
    for (size_t i = 0; i < nZ; ++i) {
        indices_[i] = rand() % cols;
        data_[i] = rand(); // & data_limit;
    }



    printf("allocating %d Mbytes of managed memory\n", sizeof(fr_t)*nZ / (1024*1024));
    err = cudaMallocManaged(&ABCZ, sizeof(fr_t)* nZ);
    if (err != cudaSuccess){ 
        printf("Error alocating memory ABCZ");
        }

    printf("allocating %d Mbytes of managed memory\n", sizeof(fr_t)*nRows_ / (1024*1024));
    err = cudaMallocManaged(&res_cuda, sizeof(fr_t)* nRows_);
    if (err != cudaSuccess){ 
        printf("Error alocating memory res_cuda");
        }

    printf("allocating %d Mbytes of managed memory\n", sizeof(fr_t)*cols / (1024*1024));
    err = cudaMallocManaged(&w_cuda, sizeof(fr_t)* cols);
    if (err != cudaSuccess){ 
        printf("Error alocating memory w_cuda");
        }

    //random witness
    for(size_t i=0; i<nCols; i++){
        wit[i] = rand();
    }
    



    printf("starting preprocessing...\n");
    auto  start = high_resolution_clock::now();
    sparseMatrixLoad<fr_t>(data_, indices_, indptr_, nZ, nRows_,
                        NULL, NULL, NULL, 0, 0,
                        NULL, NULL, NULL, 0, 0,
                        cols);
    auto  stop = high_resolution_clock::now();
    auto  duration = duration_cast<milliseconds>(stop - start);
    printf(">>preprocessing = %d ms\n", duration.count());


    start = high_resolution_clock::now();
    auto res = multiplyWitnessCPU<fr_t>(wit);
    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf(">>CPU mult = %d ms\n", duration.count());

    
    int blockSize = 64;
    int gridSize  = 512;
    int sharedMem = blockSize*sizeof(fr_t);

        start = high_resolution_clock::now();
    multiplyWitnessKernel<<<gridSize, blockSize>>>(ABCZ, w_cuda);
    CUDASYNC("Multiply");

    sumWitnessKernel<<<gridSize, blockSize, sharedMem>>>(res_cuda, ABCZ);
    CUDASYNC("Sum")
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        printf(">>GPU = %d ms\n", duration.count());


    for (int i=0; i<cols; i++){
        if (!(res_cuda[i] == res[i])){
            printf("Error detected at idx %d, further errors ignored\n");
        }
    }

    start = high_resolution_clock::now();
    
    auto resS = multiplyWitnessCPU_straight<fr_t>(wit, data_, indices_, indptr_, cols, nRows_, nZ);

    stop = high_resolution_clock::now();
    duration = duration_cast<milliseconds>(stop - start);
    printf(">>CPU mult naive = %d ms\n", duration.count());
}

int main_(){

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

    fr_t w[20] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}; // {37, 59, 0, 138, 51, 0, 11, 0, 0, 0, 87, 46, 61, 0, 140, 59, 0, 0, 214, 0}; 
    
    // sparseMatrixLoad<fr_t>(A_data, A_indices, A_indptr, A_NNZ, A_nRows,
    //                        NULL, NULL, NULL, 0, 0,
    //                        NULL, NULL, NULL, 0, 0,
    //                        nCols_l);

    sparseMatrixLoad<fr_t>(A_data, A_indices, A_indptr, A_NNZ, A_nRows,
                           A_data, A_indices, A_indptr, A_NNZ, A_nRows,
                           A_data, A_indices, A_indptr, A_NNZ, A_nRows,
                           nCols_l);



        auto start = high_resolution_clock::now();
    auto res = multiplyWitnessCPU<fr_t>(w); //[0, 427, 7959, 27401, 26184, 44341, 11623, 0, 11487, 1771]
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        printf(">>CPU = %d ms\n", duration.count());
    
    pfr(res);

    printf("GPU\n");

    #ifndef CUDASYNC
    #define CUDASYNC(fmt, ...)                                                                                             \
        err = cudaDeviceSynchronize();                                                                                     \
        if (err != cudaSuccess){                                                                                           \
        printf("\n%s:%d " fmt " Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err), ##__VA_ARGS__);           \
        }                                                                                                     
    #endif

    cudaError_t err;
    cudaMallocManaged(&ABCZ, sizeof(fr_t)* 75);
    cudaMallocManaged(&res_cuda, sizeof(fr_t)* 30);
    cudaMallocManaged(&w_cuda, sizeof(fr_t)* 20);

    for(int i=0; i<20; i++) w_cuda[i] = 1;  //unit multiplicand

    
    int blockSize = 16;
    int gridSize  = 4;
    int sharedMem = blockSize*sizeof(fr_t);
        start = high_resolution_clock::now();
    multiplyWitnessKernel<<<gridSize, blockSize>>>(ABCZ, w_cuda); //1 thread, one block.
    CUDASYNC("Multiply");

    sumWitnessKernel<<<gridSize, blockSize, sharedMem>>>(res_cuda, ABCZ);
    CUDASYNC("Sum")
        stop = high_resolution_clock::now();
        duration = duration_cast<milliseconds>(stop - start);
        printf(">>GPU = %d ms\n", duration.count());

    // printf("ABCZ m:");
    // pfr(ABCZ, A_NNZ*3);


    printf("resCuda m:");
    pfr(res_cuda, 30);


    printf("end\n");

}

