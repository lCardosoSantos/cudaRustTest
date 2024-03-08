#include <stdio.h>
#include "msm.cuh"

extern "C" void msm(g1a_t *out, const g1a_t *points, const fr_t *scalars, const size_t nPoints){
    
    printf("g1ar_t *out, const = %p\n", out);
    printf("g1ar_t *points, = %p\n", points);
    printf("const fr_t *scalars, = %p\n", scalars);
    printf("const size_t nPoints= %d\n", nPoints);

    out->x[0]= 1024;
    out->y[1] = 2048;

    return;

}



//extern C is needed so the compiler doesn't do the c++ style name mangling.

//nvcc -o clad.elf -Iinclude src/clad.cu src/fp/fp.cu src/fr/fr.cu src/g1/g1.cu
