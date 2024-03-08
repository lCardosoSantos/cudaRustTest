#pragma once

#include<stdint.h>
#include<stdio.h>
#include <cassert>
#include <cstdint>

//TODO: placeholder definitionf or fr
class fr_t {
    uint64_t _[4];

    public:

    fr_t(const fr_t &) = default;

    __host__ __device__ fr_t(uint64_t x = 0)
    {
        _[0] = x;
        _[1] = 0;
        _[2] = 0;
        _[3] = 0;
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



 __device__ __host__ void fromUint64(fr_t &z, const uint64_t *x);
 __device__ void toUint64(const fr_t &x, uint64_t *z);
 __device__ __host__ void cpy(fr_t &z, const fr_t &x);

 
inline  __device__ void reduce4(fr_t &z);
inline  __device__ void neg(fr_t &z);
inline  __device__ void x2(fr_t &z);
inline  __device__ void x3(fr_t &z);
inline  __device__ void x4(fr_t &z);
inline  __device__ void x8(fr_t &z);
inline  __device__ void x12(fr_t &z);
inline  __device__ void add(fr_t &z, const fr_t &x);
inline  __device__ void sub(fr_t &z, const fr_t &x);
inline  __device__ void addsub(fr_t &x, fr_t &y);
inline  __device__ void sqr(fr_t &z);
inline  __device__ void mul(fr_t &z, const fr_t &x);
inline  __device__ void inv(fr_t &z);
inline  __device__ __host__ void zero(fr_t &z);
inline  __device__ __host__ void one(fr_t &z);

//TODO: Should these comparison functions be inlined/ptx as well?
 __device__ bool eq(const fr_t &x, const fr_t &y);
 __device__ bool neq(const fr_t &x, const fr_t &y);
 __device__ bool nonzero(const fr_t &x);
 __device__ bool iszero(const fr_t &x);
 __device__ bool isone(const fr_t &x);

#ifdef DEBUG
 extern "C" __device__ void print(const char *s, const fr_t &x);
#endif


//Test for integration
void fr();
