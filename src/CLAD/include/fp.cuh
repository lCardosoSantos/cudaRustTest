#include<stdint.h>
#include<stdio.h>

typedef uint64_t fp_t[4]; 

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

#ifdef DEBUG
extern "C"  __device__ void print(const char *s, const fp_t &x);
#endif

//Test for integration
void fp();
