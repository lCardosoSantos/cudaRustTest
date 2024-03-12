#include "fp.cuh" 

__host__ void field_printh(const char *s, const fp_t &x, FILE *out){
    fprintf(out, "%s{fp: 0x%016lx 0x%016lx 0x%016lx 0x%016lx}\n", s, x[0], x[1], x[2], x[3]);
}

__device__ void field_print(const char *s, const fp_t &x){
    printf("%s{fp: 0x%016lx 0x%016lx 0x%016lx 0x%016lx}\n", s, x[0], x[1], x[2], x[3]);
}
