#pragma once

#include <stdint.h>
#include <stdio.h>
#include <cassert>
#include <cstdint>


class fp_t {
    private:

    uint64_t _[4];

    public:

    fp_t(const fp_t &) = default;
    ~fp_t() = default;

    // Conversions to and from internal (Montgomery) format

    __host__ __device__ fp_t(uint64_t x0 = 0, uint64_t x1 = 0, uint64_t x2 = 0, uint64_t x3 = 0);

    __host__ __device__ void to_uint64_t(uint64_t &x0, uint64_t &x1, uint64_t &x2, uint64_t &x3);

    __host__ __device__ void set_zero(){
        //temporary setter for testing
        *this = fp_t(0);
        // fp_t();
    }

    __host__ __device__ void set_one(){
        //temporary setter for testing
        *this = fp_t(1);
        // fp_t(1);
    }

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

__device__ __host__ void cpy(fp_t &z, const fp_t &x);

// Set to specific residue

__device__ __host__ void fp_set(fp_t &z, uint64_t x0 = 0, uint64_t x1 = 0, uint64_t x2 = 0, uint64_t x3 = 0);

// Arithmetic

__device__ void fp_add(fp_t &z, const fp_t &x, const fp_t &y);
__device__ void fp_sub(fp_t &z, const fp_t &x, const fp_t &y);
__device__ void fp_mul(fp_t &z, const fp_t &x, const fp_t &y);
__device__ void fp_sqr(fp_t &z, const fp_t &x);
__device__ void fp_inv(fp_t &z, const fp_t &x);
__device__ void fp_x2 (fp_t &z, const fp_t &x);
__device__ void fp_x3 (fp_t &z, const fp_t &x);
__device__ void fp_x4 (fp_t &z, const fp_t &x);
__device__ void fp_x8 (fp_t &z, const fp_t &x);
__device__ void fp_x12(fp_t &z, const fp_t &x);
__device__ void fp_neg(fp_t &z, const fp_t &x);

// Comparisons

__device__ __host__ bool fp_eq(const fp_t &x, const fp_t &y);
__device__ __host__ bool fp_ne(const fp_t &x, const fp_t &y);

__device__ __host__ bool fp_is0(const fp_t &x);
__device__ __host__ bool fp_is1(const fp_t &x);

namespace fp {

    // Overloaded functions

    inline __device__ void add(fp_t &z, const fp_t &x, const fp_t &y) { fp_add(z, x, y); }
    inline __device__ void sub(fp_t &z, const fp_t &x, const fp_t &y) { fp_sub(z, x, y); }
    inline __device__ void mul(fp_t &z, const fp_t &x, const fp_t &y) { fp_mul(z, x, y); }
    inline __device__ void sqr(fp_t &z, const fp_t &x) { fp_mul(z, x, x); }
    inline __device__ void inv(fp_t &z, const fp_t &x) { fp_inv(z, x); }
    inline __device__ void x2 (fp_t &z, const fp_t &x) { fp_x2 (z, x); }
    inline __device__ void x3 (fp_t &z, const fp_t &x) { fp_x3 (z, x); }
    inline __device__ void x4 (fp_t &z, const fp_t &x) { fp_x4 (z, x); }
    inline __device__ void x8 (fp_t &z, const fp_t &x) { fp_x8 (z, x); }
    inline __device__ void x12(fp_t &z, const fp_t &x) { fp_x12(z, x); }
    inline __device__ void neg(fp_t &z, const fp_t &x) { fp_neg(z, x); }

    inline __device__ __host__ bool eq(const fp_t &x, const fp_t &y) { fp_eq(x, y); }
    inline __device__ __host__ bool ne(const fp_t &x, const fp_t &y) { fp_ne(x, y); }

    inline __device__ __host__ bool operator==(const fp_t &x, const fp_t &y) { fp_eq(x, y); }
    inline __device__ __host__ bool operator!=(const fp_t &x, const fp_t &y) { fp_ne(x, y); }

    inline __device__ __host__ bool is0(const fp_t &x) { fp_is0(x); }
    inline __device__ __host__ bool is1(const fp_t &x) { fp_is1(x); }

};


// #ifdef DEBUG
__host__   void field_printh(const char *s, const fp_t &x, FILE *out = stdout);
__device__ void field_print(const char *s, const fp_t &x);
// #endif
