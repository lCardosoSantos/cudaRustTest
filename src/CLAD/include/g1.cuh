#pragma once
#pragma diag_suppress 4

#include<stdio.h>
#include "fp.cuh"
#include "fr.cuh"

// A pair of residues representing a G1 point in affine coordinates

struct g1a_t {
    fp_t x, y;

    __host__ __device__ g1a_t(const fp_t &x, const fp_t &y)
    : x(x)
    , y(y)
    {}

    __host__ __device__ g1a_t()
    {
        x.set_zero();
        y.set_zero();
    }
};

// A triple of residues representing a G1 point in homogeneous projective coordinates

struct g1p_t {
    fp_t x, y, z;

    __host__ __device__ g1p_t()
    {
        x.set_zero();
        y.set_one();
        z.set_zero();
    }

    __host__ __device__ g1p_t(const fp_t &x, const fp_t &y, const fp_t &z)
    : x(x)
    , y(y)
    , z(z)
    {}

    __host__ __device__ g1p_t(const g1a_t &a)
    : x(a.x)
    , y(a.y)
    {
        z.set_one();

        if (fp_is0(x) && fp_is0(y))
        {
            y.set_one();
            z.set_zero();
        }
    }
};


__device__ __host__ void g1a_fromUint64(g1a_t &a, const uint64_t *x, const uint64_t *y);
__device__ __host__ void g1a_fromFp(g1a_t &a, const fp_t &x, const fp_t &y);
__device__ void          g1a_fromG1p(g1a_t &a, const g1p_t &p);
__device__ __host__ void g1a_cpy(g1a_t &a, const g1a_t &b);

__device__ void          g1p_toUint64(const g1p_t &p, uint64_t *x, uint64_t *y, uint64_t *z);
__device__ __host__ void g1p_fromUint64(g1p_t &p, const uint64_t *x, const uint64_t *y, const uint64_t *z);
__device__ __host__ void g1p_fromFp(g1p_t &p, fp_t &x, fp_t &y, fp_t &z);
__device__ void          g1p_fromG1a(g1p_t &p, const g1a_t &a);
__device__ __host__ void g1p_cpy(g1p_t &p, const g1p_t &q);

__device__ bool g1p_eq(const g1p_t &p, const g1p_t &q);
__device__ bool g1p_neq(const g1p_t &p, const g1p_t &q);
__device__ bool g1p_isInf(const g1p_t &p);
__device__ bool g1p_isPoint(const g1p_t &p);

__device__ void g1p_neg(g1p_t &p);
__device__ void g1p_scale(g1p_t &p, const fp_t &s);

__device__ void g1p_sub   (g1p_t &a, const g1p_t &x, const g1p_t &y);
__device__ void g1p_mul   (g1p_t &a, const g1p_t &x, const fr_t &y);
__device__ void g1p_dbl   (g1p_t &a, const g1p_t &x);
__device__ void g1p_add   (g1p_t &a, const g1p_t &x, const g1p_t &y);
__device__ void g1p_addsub(g1p_t &p, g1p_t &q); //TODO: Maybe it is more usefull as g1p_addsub(a, b, p, q) -> a= p+q; b=p-q


// PTX acelerator
// __device__ void g1p_multi(int op, g1p_t *p, g1p_t *q, const g1p_t *r, const g1p_t *s);
// inline __device__ void g1p_dbl(g1p_t &p) { g1p_multi(-1, &p, nullptr, &p, &p); }
// inline __device__ void g1p_add(g1p_t &p, const g1p_t &q) { g1p_multi(-2, &p, nullptr, &p, &q); }

__device__ __host__ void g1a_inf(g1a_t &a);
__device__ __host__ void g1a_gen(g1a_t &a);

__device__ __host__ void g1p_inf(g1p_t &p);
__device__ __host__ void g1p_gen(g1p_t &p);

// #ifdef DEBUG
__host__ void g1p_printh(const char *s, const g1p_t &p, FILE *out = stdout);
__device__ void g1p_print(const char *s, const g1p_t &p);


__host__ void g1a_printh(const char *s, const g1a_t &a, FILE *out = stdout);
__device__ void g1a_print(const char *s, const g1a_t &a);
// #endif

// //test for integration
// void g1();
