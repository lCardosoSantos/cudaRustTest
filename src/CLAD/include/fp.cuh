#pragma once

#include<stdint.h>
#include<stdio.h>
#include <cassert>
#include <cstdint>


class fp_t {
    uint64_t _[4];

    public:

    fp_t(const fp_t &) = default;

    __host__ __device__ fp_t(uint64_t x = 0)
    {
        _[0] = x;
        _[1] = 0;
        _[2] = 0;
        _[3] = 0;
    }

    __host__ __device__ fp_t(uint64_t x, uint64_t y, uint64_t z, uint64_t a)
    {
        _[0] = x;
        _[1] = y;
        _[2] = z;
        _[3] = a;
    }

    __host__ __device__ uint64_t &operator[](int i)
    {
        assert(0 <= i);
        assert(i <= 3);
        return _[i];
    }

    __host__ __device__ uint64_t  operator[](int i) const
    {
        assert(0 <= i);
        assert(i <= 3);
        return _[i];
    }
};

__device__ __host__ void fromUint64(fp_t &z, const uint64_t *x);
__device__ __host__ void toUint64(uint64_t *z, const fp_t &x);
__device__ __host__ void cpy(fp_t &z, const fp_t &x);


inline  __device__ void reduce6(fp_t &z);
inline  __device__ void neg(fp_t &z, const fp_t &x);
inline  __device__ void x2(fp_t &z, const fp_t &x);
inline  __device__ void x3(fp_t &z, const fp_t &x);
inline  __device__ void x4(fp_t &z, const fp_t &x);
inline  __device__ void x8(fp_t &z, const fp_t &x);
inline  __device__ void x12(fp_t &z, const fp_t &x);
inline  __device__ void add(fp_t &z, const fp_t &x, const fp_t &y);
inline  __device__ void sub(fp_t &z, const fp_t &x, const fp_t &y);
inline  __device__ void sqr(fp_t &z, const fp_t &x);
inline  __device__ void mul(fp_t &z, const fp_t &x, const fp_t &y);
inline  __device__ void mma(fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y);
inline  __device__ void inv(fp_t &z, const fp_t &x);
__device__ __host__ void zero(fp_t &z);
__device__ __host__ void one(fp_t &z);

//TODO: Should these comparison functions be inlined/ptx as well?
__device__ bool eq(const fp_t &x, const fp_t &y);
__device__ bool neq(const fp_t &x, const fp_t &y);
__device__ bool nonzero(const fp_t &x);
__device__ bool iszero(const fp_t &x);
__device__ bool isone(const fp_t &x);

// #ifdef DEBUG
__host__   void field_printh(const char *s, const fp_t &x, FILE *out = stdout);
__device__ void field_print(const char *s, const fp_t &x);
// #endif

//Test for integration
void fp();
