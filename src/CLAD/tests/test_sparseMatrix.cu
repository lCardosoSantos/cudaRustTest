#include "testUtil.cuh"
#include "SparseMatrix.cuh"

//Simplistic test to check build
//TODO: More in depth test
extern "C" void run_sparseMatrix_test(){
    stdout_isatty = isatty(fileno(stdout)); 

    printf("\nSparse Matrix tests");
    
    //[1, 2, 0, 0, 0, 0]
    //[0, 3, 0, 4, 0, 0]
    //[0, 0, 5, 6, 7, 0]
    //[0, 0, 0, 0, 0, 8]
    int data[] = {1, 2, 3, 4, 5, 6, 7, 8}; 
    size_t indices[] = {0, 1, 1, 3, 2, 3, 4, 5};
    size_t indptr[] =  {0, 2, 4, 5, 8};
    size_t rows = 4;
    size_t cols = 6;

    SparseMatrix_t<int> m(&data[0], &indices[0], &indptr[0], rows, cols, 8);

    if(m.get(2, 2) != 5){
        PRINTPASS(false);
    }
    else{
        PRINTPASS(true);
    }

}
