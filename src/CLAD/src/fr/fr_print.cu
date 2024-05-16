// Copyright 2022-2024 Dag Arne Osvik
// Copyright 2022-2024 Luan Cardoso dos Santos
#include "fr.cuh" 

__host__ void field_printh(const char *s, const fr_t &x, FILE *out){
    fprintf(out, "%s{fr: 0x%016lx 0x%016lx 0x%016lx 0x%016lx}\n", s, x[0], x[1], x[2], x[3]);
}

__device__ void field_print(const char *s, const fr_t &x){
    printf("%s{fr: 0x%016lx 0x%016lx 0x%016lx 0x%016lx}\n", s, x[0], x[1], x[2], x[3]);
}
