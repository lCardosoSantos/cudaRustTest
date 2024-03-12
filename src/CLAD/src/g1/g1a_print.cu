#include "g1.cuh" 

__host__ void g1a_printh(const char *s, const g1a_t &a, FILE *out){
    fprintf(out, "%s{g1:\n", s);
    field_printh("\tx:", a.x, out);
    field_printh("\ty:", a.y, out);
    fprintf(out, "}\n");
}


__device__ void g1a_print(const char *s, const g1a_t &a){
    printf("%s{g1:\n", s);
    field_print("\tx:", a.x);
    field_print("\ty:", a.y);
    printf("}\n");
}
