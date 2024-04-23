#include <stdio.h>
#include <assert.h>
#include "fp.cuh"
#include "SparseMatrix.cuh"
#include "matrixMult.cuh"

SparseMatrix_t<fp_t> *A, *B, *C; 

extern "C"
void matrix_load(void *data, size_t *indices, size_t *indptr, 
                size_t nElements, size_t cols, size_t rows, int matrixIndex){

    SparseMatrix_t tmp = SparseMatrix_t<fp_t>((fp_t*)data, indices, indptr,
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
__global__ void multiplyWitness(field *AZ, field *BZ, field *Cz, field *witness){
    
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

    multiplyWitness<fp_t><<<NTHREADS, NBLOCKS>>>((fp_t*)AZ, (fp_t*)BZ, (fp_t*)CZ, 
                                                 (fp_t*)witness);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess){                         
        printf("\n%s:%d  Error: %d (%s)\n", __FILE__, __LINE__, err, cudaGetErrorName(err));
    }

    return;
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
