#include "g1.cuh" 
#include "fp.cuh"

__host__ void g1p_printh(const char *s, const g1p_t &p, FILE *out){
    fprintf(out, "%s{g1:\n", s);
    field_printh("\tx:", p.x, out);
    field_printh("\ty:", p.y, out);
    field_printh("\tz:", p.z, out);
    fprintf(out, "}\n");
}

__device__ void g1p_print(const char *s, const g1p_t &p){
    printf("%s{g1:\n", s);
    field_print("\tx:", p.x);
    field_print("\ty:", p.y);
    field_print("\tz:", p.z);
    printf("}\n");
}
