#include "fr.cuh"
#include "fp.cuh"
#include "g1.cuh"

#include<stdio.h>
void clad(){
    printf("CLAD\n");
    fp();
    fr();
    g1();
}

// int main(){
//     clad();
// }

//nvcc -o clad.elf -Iinclude src/clad.cu src/fp/fp.cu src/fr/fr.cu src/g1/g1.cu
