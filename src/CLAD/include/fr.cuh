#include<stdint.h>
#include<stdio.h>

//TODO: placeholder definitionf or fp
typedef uint64_t fr_t[4]; 

extern "C" __device__ __host__ void fr_fromUint64(fr_t &z, const uint64_t *x);
extern "C" __device__ void fr_toUint64(const fr_t &x, uint64_t *z);
extern "C" __device__ __host__ void fr_cpy(fr_t &z, const fr_t &x);
extern "C" __device__ void fr_reduce4(fr_t &z);
extern "C" __device__ void fr_neg(fr_t &z);
extern "C" __device__ void fr_x2(fr_t &z);
extern "C" __device__ void fr_x3(fr_t &z);
extern "C" __device__ void fr_x4(fr_t &z);
extern "C" __device__ void fr_x8(fr_t &z);
extern "C" __device__ void fr_x12(fr_t &z);
extern "C" __device__ void fr_add(fr_t &z, const fr_t &x);
extern "C" __device__ void fr_sub(fr_t &z, const fr_t &x);
extern "C" __device__ void fr_addsub(fr_t &x, fr_t &y);
extern "C" __device__ void fr_sqr(fr_t &z);
extern "C" __device__ void fr_mul(fr_t &z, const fr_t &x);
extern "C" __device__ void fr_inv(fr_t &z);
extern "C" __device__ __host__ void fr_zero(fr_t &z);
extern "C" __device__ __host__ void fr_one(fr_t &z);

extern "C" __device__ bool fr_eq(const fr_t &x, const fr_t &y);
extern "C" __device__ bool fr_neq(const fr_t &x, const fr_t &y);
extern "C" __device__ bool fr_nonzero(const fr_t &x);
extern "C" __device__ bool fr_iszero(const fr_t &x);
extern "C" __device__ bool fr_isone(const fr_t &x);

#ifdef DEBUG
extern "C" __device__ void fr_print(const char *s, const fr_t &x);
#endif


//Test for integration
void fr();
