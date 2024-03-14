#pragma once

#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <cstdint>


class fr_t {
    private:

    uint64_t _[4];

    public:

    fr_t(const fr_t &) = default;
    ~fr_t() = default;

    // Conversions to and from internal (Montgomery) format

    __host__ __device__ fr_t(uint64_t x0 = 0, uint64_t x1 = 0, uint64_t x2 = 0, uint64_t x3 = 0);

    __host__ __device__ void to_uint64_t(uint64_t &x0, uint64_t &x1, uint64_t &x2, uint64_t &x3);

    // Direct access to internal representation, for implementation convenience.

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

__device__ __host__ void cpy(fr_t &z, const fr_t &x);

// Set to specific residue

__device__ void fr_set(fr_t &z, uint64_t x0 = 0, uint64_t x1 = 0, uint64_t x2 = 0, uint64_t x3 = 0);

// Arithmetic

__device__ void fr_add(fr_t &z, const fr_t &x, const fr_t &y);
__device__ void fr_sub(fr_t &z, const fr_t &x, const fr_t &y);
__device__ void fr_mul(fr_t &z, const fr_t &x, const fr_t &y);
__device__ void fr_sqr(fr_t &z, const fr_t &x);
__device__ void fr_inv(fr_t &z, const fr_t &x);
__device__ void fr_x2 (fr_t &z, const fr_t &x);
__device__ void fr_x3 (fr_t &z, const fr_t &x);
__device__ void fr_x4 (fr_t &z, const fr_t &x);
__device__ void fr_x8 (fr_t &z, const fr_t &x);
__device__ void fr_x12(fr_t &z, const fr_t &x);
__device__ void fr_neg(fr_t &z, const fr_t &x);

// Comparisons

__device__ bool fr_eq(const fr_t &x, const fr_t &y);
__device__ bool fr_ne(const fr_t &x, const fr_t &y);

__device__ bool fr_is0(const fr_t &x);
__device__ bool fr_is1(const fr_t &x);

namespace fr {

    // Overloaded functions

    inline __device__ void add(fr_t &z, const fr_t &x, const fr_t &y) { fr_add(z, x, y); }
    inline __device__ void sub(fr_t &z, const fr_t &x, const fr_t &y) { fr_sub(z, x, y); }
    inline __device__ void mul(fr_t &z, const fr_t &x, const fr_t &y) { fr_mul(z, x, y); }
    inline __device__ void sqr(fr_t &z, const fr_t &x) { fr_mul(z, x, x); }
    inline __device__ void inv(fr_t &z, const fr_t &x) { fr_inv(z, x); }
    inline __device__ void x2 (fr_t &z, const fr_t &x) { fr_x2 (z, x); }
    inline __device__ void x3 (fr_t &z, const fr_t &x) { fr_x3 (z, x); }
    inline __device__ void x4 (fr_t &z, const fr_t &x) { fr_x4 (z, x); }
    inline __device__ void x8 (fr_t &z, const fr_t &x) { fr_x8 (z, x); }
    inline __device__ void x12(fr_t &z, const fr_t &x) { fr_x12(z, x); }
    inline __device__ void neg(fr_t &z, const fr_t &x) { fr_neg(z, x); }

    inline __device__ bool eq(const fr_t &x, const fr_t &y) { fr_eq(x, y); }
    inline __device__ bool ne(const fr_t &x, const fr_t &y) { fr_ne(x, y); }

    inline __device__ bool operator==(const fr_t &x, const fr_t &y) { fr_eq(x, y); }
    inline __device__ bool operator!=(const fr_t &x, const fr_t &y) { fr_ne(x, y); }

    inline __device__ bool is0(const fr_t &x) { fr_is0(x); }
    inline __device__ bool is1(const fr_t &x) { fr_is1(x); }

};


// #ifdef DEBUG
__host__   void field_printh(const char *s, const fr_t &x, FILE *out = stdout);
__device__ void field_print(const char *s, const fr_t &x);
// #endif
