#include <stdio.h>
// #include "g1.cuh"
#include "fp.cuh"

extern "C" void scratchboard_cuda(){
    printf("Scratchboard: For testing build while the makefiles are not written\n\n");

    fp_t a;
    a[2]=0xcafecafe;

    field_printh("Test", a);
    a.set_one();
    field_printh("one", a);
    a.set_zero();
    field_printh("zero", a);


    #ifdef RUST_TEST
        printf("RUST_TEST is defined!\n");
    #endif

    printf("---\n\n");


}


/*

cargo run  2>/dev/null
rm .buildlog; cargo build --verbose &> .buildlog; code .buildlog

*/
