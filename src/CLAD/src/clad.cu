#include "clad.cuh"
#include<stdio.h>

extern "C" void clad(){
    printf("CLAD\n");
    fp();
    fr();
    g1();
}

extern "C" void clad_msm(g1a_t *out, const g1a_t *points, const fr_t *scalars, const size_t nPoints){

    return;
}



//extern C is needed so the compiler doesn't do the c++ style name mangling.

//nvcc -o clad.elf -Iinclude src/clad.cu src/fp/fp.cu src/fr/fr.cu src/g1/g1.cu
