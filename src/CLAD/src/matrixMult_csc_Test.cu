/*
For testing an alternative way of executing the matrix-witness multiplicaiton.

# Preprocessing step:
    A, B and C come from rust in CSR format.
    A||B||C is copied to managed memory, being converted to CSC format.
    updated rowptr is copied to managed memory
    colptr is generated for this new matrix
    An array of fr_t[NNZ] is allocated in managed memory. When data is copied to
        managed memory, an array of pointers is created, using the original row 
        position, and it is used as index to where the result of the multiplication
        is written to.

# Multiplication kernel
    Each block runs the multiplication for each column, using colptr. 
    The block loads the multiplicand from witness to shared memory.
    iterating over the column, every thread loads data to shared memory (avoiding bank conflicts), and loads the writing pointers
        (writing location could be saved on registers)
    Executes multiplication and writes to destination
        fr=32 bytes, pointer = 8 bytes, bank = 32bits * 32banks, warp = 32threads

#sum kernel
    each block is responsible for summing a row, using rowptr.
    the block iterates over the array, loads elements into shared memory, and sums them to a local accumulator.
    at the end, threads gather_sum and writes to destination = rowIndex.
    
#pitfalls
    Data processed by each block is not multiple of block size. Last iteration will have some inactive threads.


    Matrices use 5250mb for data, plus extra for indices.
    5250mb extra for the temporaries

    - data
    - multiplcationWriteLocationPointer
    - colptr
    - rowptr

    - tmp for multiplication
*/

//needed
#include <stdio.h>
#include <stdint.h>

//debug extra
#include <vector>
#include <time.h>
#include<unistd.h>
#include <chrono>

using namespace std;
using std::vector; 
using Clock = std::chrono::steady_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;


////////////////////////////////////////////////////////////////////////////////
//                            Debug helper                                    //
////////////////////////////////////////////////////////////////////////////////

#ifdef DEBUG
    #define DPRINTF(fmt, ...) fprintf(stderr, "[debug] %s:%d " fmt "\n", __FILE__, __LINE__,  ##__VA_ARGS__)
#else
    #define DPRINTF(fmt, ...)
#endif

#define CPU_TIMER_INIT time_point<Clock> start, end
#define CPU_TIMER_START start = Clock::now()
#define CPU_TIMER_END end = Clock::now()
#define CPU_TIMER_PRINT(msg) printf("%s : %ld ms\n", msg,  duration_cast<milliseconds>(end-start).count());



class fr_t {
    //Dummy data type to replace fr_t during the tests, so compilation and debugg is easier
    public:
    uint64_t _[4];
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
        _[1] = _[0];
        _[2] = _[0];
        _[3] = _[0];

        return *this;
    }

    __host__ __device__ void mul(fr_t &output, fr_t &in1, fr_t &in2){
        output = fr_t(in1._[0] * in2._[0]);
    }

    __host__ __device__ void add(fr_t &output, fr_t &in1, fr_t &in2){
        output = fr_t(in1._[0] + in2._[0]);
    }

};

__host__ __device__ void fr_mul(fr_t &z, const fr_t &x, const fr_t &y){
    z = fr_t(x._[0]*y._[0]);
}

__host__ __device__ void fr_add(fr_t &z, const fr_t &x, const fr_t &y){
    z = fr_t(x._[0]+y._[0]);
}

//less ugly print in gdb
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

void printCSRMatrix(fr_t *data, size_t *colidx, size_t *row_ptr, size_t nrows, size_t ncols) {
    for(size_t i=0; i<nrows; i++){
        int start = row_ptr[i];
        int end = row_ptr[i+1];

        vector<uint64_t> line(ncols, 0);
        for(int j=start; j<end; j++){
            line[colidx[j]] = data[j]._[0];
        }

        for(const uint64_t& d: line)
            printf("%ld,\t", d);
        printf("\n");
    }
    printf("\n");

}

void printCSCMatrix(fr_t *data, size_t *rowidx, size_t *colptr, size_t nrows, size_t ncols) {
    vector<vector<int>> dt(nrows, vector<int>(ncols, 0)); //[row][colum]
    //fill
    for(int i=0; i<ncols; i++){
        int start = colptr[i];
        int end = colptr[i+1];

        for(int j=start; j<end; j++){
            dt[rowidx[j]][i] = data[j]._[0];
        } 
    }
    for(const auto &r: dt){
        for(const auto &i: r)
            printf("%d,\t", i);
        printf("\n");
    }
    printf("\n");

}

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
        data[i] = rand() % 255 +1; // limits data for easy mul
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




////////////////////////////////////////////////////////////////////////////////
//                                CORE                                        //
////////////////////////////////////////////////////////////////////////////////
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
    //TODO: Pointer is wrong, writing to CSC location, not CSR
    for(size_t i=0; i<A_nRows; i++){
        for(size_t j=A_indptr[i]; j<A_indptr[i+1]; j++){
            size_t col = A_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i;
            cuda_data[dest] = A_data[j];
            MultiplicationDestinationPointer[j] = &(cuda_tmp_mul[j]);
            tmp[col]++;
        }
    }

    for(size_t i=0; i<B_nRows; i++){
        for(size_t j=B_indptr[i]; j<B_indptr[i+1]; j++){
            size_t col = B_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i+A_nRows;
            cuda_data[dest] = B_data[j];
            tmp[col]++;
            MultiplicationDestinationPointer[j + A_NNZ] = &(cuda_tmp_mul[j + A_NNZ]);
        }
    }

    for(size_t i=0; i<C_nRows; i++){
        for(size_t j=C_indptr[i]; j<C_indptr[i+1]; j++){
            size_t col = C_colidx[j];
            size_t dest = tmp[col];

            cuda_rowidx[dest] = i+A_nRows+B_nRows;
            cuda_data[dest] = C_data[j];
            tmp[col]++;
            MultiplicationDestinationPointer[j + A_NNZ + B_NNZ] = &(cuda_tmp_mul[j + A_NNZ + B_NNZ]);
        }
    }

    //// update rowptr

    for(size_t i=0; i<=A_nRows; i++)
        cuda_rowptr[i] = A_indptr[i];

    for(size_t i=0; i<=B_nRows; i++)
        cuda_rowptr[i+A_nRows] = B_indptr[i] + A_nRows;

    for(size_t i=0; i<=C_nRows; i++)
        cuda_rowptr[i+A_nRows+B_nRows] = C_indptr[i] + A_nRows + B_nRows;

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
        for(int j=0; j<8; j++){
            uint32_t *mpointer = (uint32_t *)(&cuda_data[i+tidx]);
            M[(10*tidx)+j] = mpointer[j];
        }
        __syncthreads();

        //multiply and write to destination
        T *sharedM = (T*)(&M[10*tidx]);
        fr_mul(*MultiplicationDestinationPointer[i], *sharedM, *wit);


    }
    
    //multiplication tail
    if (tidx < end-mend){
        //load data into shared memory
        for(int j=0; j<8; j++){
            uint32_t *mpointer = (uint32_t *)(&cuda_data[mend+tidx]);
            M[(10*tidx)+j] = mpointer[j];
        }
        __syncthreads();
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
    unsigned gdim = gridDim.x; 

    //each block operates on a single row

    //init local accumulator to zero

    //loop over row data, loading one blok into shared memory up to a multiple of block size
    //(better to load to register space?)

    //accumulate.

    //load and accumulate tail

    //reduce sum

    //thread zero writes to destination

}

template<typename T>
void multiplyWitness(T *res, T *witness){
    //wrapper for the kernel calling
    const size_t blockSize = 32; //number of threads 
    const int sharedMem = (blockSize+1) * (sizeof(T)+sizeof(uint64_t)); //dynamic shared mem
    const size_t gridSizeMul = cuda_nCols;
    const size_t gridSizeAdd = cuda_nRows;
    cudaError_t err;

    //check if matrices and memory is allocated.
    if(!cuda_matrix_multiplication_ready) return; 


    // CPU_TIMER_INIT;
    
    // CPU_TIMER_START;
    multiplyWitnessKernel<fr_t><<<gridSizeMul, blockSize, sharedMem>>>(witness);
    CUDASYNC("Multiplication Kernel");
    // multiplyWitnessKernel<fr_t><<<gridSizeAdd, blockSize, sharedMem>>>(res);
    // CUDASYNC("Sum Kernel");
    // CPU_TIMER_END;
    // CPU_TIMER_PRINT("Gpu kernel multiplication ");


}

////////////////////////////////////////////////////////////////////////////////
//                                 MAIN                                       //
////////////////////////////////////////////////////////////////////////////////
__managed__ fr_t *witness, *resCuda;
void testKernels(){
    CPU_TIMER_INIT;
#ifdef BIG
    #warning "Compiling for ~7GB data"
    static size_t aNNZ   = 104675466;//10;
    static size_t bNNZ   = 57990791;//10;
    static size_t cNNZ   = 9101902;//10;
    static size_t anRows = 9825045;//30;
    static size_t bnRows = 9825045;//20;
    static size_t cnRows = 9825045;//10;
    static size_t ncols  = 7999846;//20;
#else
    static size_t aNNZ   = 10;
    static size_t bNNZ   = 10;
    static size_t cNNZ   = 10;
    static size_t anRows = 30;
    static size_t bnRows = 20;
    static size_t cnRows = 10;
    static size_t ncols  = 70;
#endif

    size_t abcnrows=anRows+bnRows+cnRows, abcNNZ=aNNZ+bNNZ+cNNZ;

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
    sparseMatrixLoad<fr_t, size_t>(adata, acolidx, arowptr, aNNZ, anRows, 
                                   bdata, bcolidx, browptr, bNNZ, bnRows, 
                                   cdata, ccolidx, crowptr, cNNZ, cnRows, 
                                   ncols);
    CPU_TIMER_END;
    CPU_TIMER_PRINT("Matrix Load:")

    
    //easyWitness for debug

    cudaMallocManaged(&witness, ncols*sizeof(fr_t));    
    for(int i=0; i<ncols; i++) witness[i]=fr_t(i);
    
    cudaMallocManaged(&resCuda, cuda_nRows*sizeof(fr_t));


    CPU_TIMER_START;
    auto resCPU = multiplyWitnessCPU_CSC(witness, cuda_data, cuda_rowidx, cuda_colptr, ncols, abcnrows, abcNNZ);
    CPU_TIMER_END;
    CPU_TIMER_PRINT("CPU matrixMult:")

   #ifndef BIG 

    printf("ABC:\n");
    printCSCMatrix(cuda_data, cuda_rowidx, cuda_colptr, cuda_nRows, cuda_nCols);
    printf("\n\n");

    printf("resCPU:\n");
    for(int i=0; i<abcnrows; i++) printf("%lu, ", resCPU[i]._[0]);
    printf("\n\n");

   #endif

    CPU_TIMER_START;
    multiplyWitness(resCuda, witness);
    CPU_TIMER_END;
    CPU_TIMER_PRINT("GPU matrixMult:")

   #ifndef BIG 

    printf("resCUDA:\n");
    for(int i=0; i<abcnrows; i++) printf("%lu, ", resCuda[i]._[0]);
    printf("\n\n");

    printf("cuda_data:\n");
    for(int i=0; i<cuda_NNZ; i++) printf("%lu, ", cuda_data[i]._[0]);
    printf("\n\n");

    printf("cuda_tmp_mul:\n");
    for(int i=0; i<cuda_NNZ; i++) printf("%lu, ", cuda_tmp_mul[i]._[0]);
    printf("\n\n");
   #endif
}

void testSparseMatrixLoad(){
    printf("testing sparse matrix load fr\n");

    size_t aNNZ = 10, aCols = 20, aRows=20;
    size_t bNNZ = 10, bCols = 20, bRows=20;
    size_t cNNZ = 10, cCols = 20, cRows=20;

    fr_t data_a[aNNZ], data_b[bNNZ], data_c[cNNZ];
    size_t colidx_a[aCols], colidx_b[bCols], colidx_c[cCols];
    size_t rowptr_a[aRows+1], rowptr_b[bRows+1], rowptr_c[cRows+1];


    srand(0xcafe);
    genRandomCSR<fr_t, size_t>(data_a, colidx_a, rowptr_a, aCols, aRows, aNNZ);
    genRandomCSR<fr_t, size_t>(data_b, colidx_b, rowptr_b, bCols, bRows, bNNZ);
    genRandomCSR<fr_t, size_t>(data_c, colidx_c, rowptr_c, cCols, cRows, cNNZ);

    printf("A:\n");
    printCSRMatrix(data_a, colidx_a, rowptr_a, aRows, aCols);
    printf("B:\n");
    printCSRMatrix(data_b, colidx_b, rowptr_b, bRows, bCols);
    printf("C:\n");
    printCSRMatrix(data_c, colidx_c, rowptr_c, cRows, cCols);
    

    sparseMatrixLoad<fr_t, size_t>(data_a, colidx_a, rowptr_a, aNNZ, aRows, 
                                   data_b, colidx_b, rowptr_b, bNNZ, bRows, 
                                   data_c, colidx_c, rowptr_c, cNNZ, cRows, 
                                   aCols);

    printf("ABC:\n");
    printCSCMatrix(cuda_data, cuda_rowidx, cuda_colptr, cuda_nRows, cuda_nCols);


}

int main(){
    // testSparseMatrixLoad();
    testKernels();
}
// void testTimer(){
//     CPU_TIMER_INIT; 
//     CPU_TIMER_START;
//     sleep(2);
//     CPU_TIMER_END;
//     CPU_TIMER_PRINT("time was");
// }
