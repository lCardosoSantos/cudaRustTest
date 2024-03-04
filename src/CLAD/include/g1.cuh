#include<stdio.h>
#include "fp.cuh"
#include "fr.cuh"

//efinition g1 affine
typedef struct {
    fp_t x, y;
} g1a_t;

//Tefinition g1 jacobian
typedef struct {
    fp_t x, y, z;
} g1p_t;

extern "C" __device__ __host__ void g1a_fromUint64(g1a_t &a, const uint64_t *x, const uint64_t *y);
extern "C" __device__ __host__ void g1a_fromFp(g1a_t &a, const fp_t &x, const fp_t &y);
extern "C" __device__ void          g1a_fromG1p(g1a_t &a, const g1p_t &p);
extern "C" __device__ __host__ void g1a_cpy(g1a_t &a, const g1a_t &b);

extern "C" __device__ void          g1p_toUint64(const g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z);
extern "C" __device__ __host__ void g1p_fromUint64(g1p_t &p, const uint64_t *x, const uint64_t *y, const uint64_t *z);
    inline __device__ __host__ void g1p_fromFp(g1p_t &p, fp_t &x, fp_t &y, fp_t &z) { g1p_fromUint64(p, x, y, z); }
extern "C" __device__ void          g1p_fromG1a(g1p_t &p, const g1a_t &a);
extern "C" __device__ __host__ void g1p_cpy(g1p_t &p, const g1p_t &q);

extern "C" __device__ bool g1p_eq(const g1p_t &p, const g1p_t &q);
extern "C" __device__ bool g1p_neq(const g1p_t &p, const g1p_t &q);
extern "C" __device__ bool g1p_isInf(const g1p_t &p);
extern "C" __device__ bool g1p_isPoint(const g1p_t &p);

extern "C" __device__ void g1p_neg(g1p_t &p);
extern "C" __device__ void g1p_scale(g1p_t &p, const fp_t &s);

extern "C" __device__ void g1p_sub(g1p_t &p, const g1p_t &q);
extern "C" __device__ void g1p_mul(g1p_t &p, const fr_t &x);
extern "C" __device__ void g1p_add(g1p_t &p, const g1a_t &q);
extern "C" __device__ void g1p_addsub(g1p_t &p, g1p_t &q);


// PTX acelerator
// extern "C" __device__ void g1p_multi(int op, g1p_t *p, g1p_t *q, const g1p_t *r, const g1p_t *s);
// inline __device__ void g1p_dbl(g1p_t &p) { g1p_multi(-1, &p, nullptr, &p, &p); }
// inline __device__ void g1p_add(g1p_t &p, const g1p_t &q) { g1p_multi(-2, &p, nullptr, &p, &q); }

extern "C" __device__ __host__ void g1a_inf(g1a_t &a);
extern "C" __device__ __host__ void g1a_gen(g1a_t &a);

extern "C" __device__ __host__ void g1p_inf(g1p_t &p);
extern "C" __device__ __host__ void g1p_gen(g1p_t &p);

#ifdef DEBUG
extern "C" __device__ __host__ void g1p_print(const char *s, const g1p_t &p);
extern "C" __device__ __host__ void g1a_print(const char *s, const g1a_t &a);
#endif

//test for integration
void g1();
