#include<stdint.h>
#include<stdio.h>

typedef uint64_t fp_t[4]; 

extern "C" __device__ __host__ void fp_fromUint64(fp_t &z, const uint64_t *x);
extern "C" __device__ __host__ void fp_toUint64(uint64_t *z, const fp_t &x);
extern "C" __device__ __host__ void fp_cpy(fp_t &z, const fp_t &x);


extern "C" __device__ void fp_reduce6(fp_t &z);
extern "C" __device__ void fp_neg(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_x2(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_x3(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_x4(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_x8(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_x12(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_add(fp_t &z, const fp_t &x, const fp_t &y);
extern "C" __device__ void fp_sub(fp_t &z, const fp_t &x, const fp_t &y);
extern "C" __device__ void fp_sqr(fp_t &z, const fp_t &x);
extern "C" __device__ void fp_mul(fp_t &z, const fp_t &x, const fp_t &y);
extern "C" __device__ void fp_mma(fp_t &z, const fp_t &v, const fp_t &w, const fp_t &x, const fp_t &y);
extern "C" __device__ void fp_inv(fp_t &z, const fp_t &x);
extern "C" __device__ __host__ void fp_zero(fp_t &z);
extern "C" __device__ __host__ void fp_one(fp_t &z);

extern "C" __device__ bool fp_eq(const fp_t &x, const fp_t &y);
extern "C" __device__ bool fp_neq(const fp_t &x, const fp_t &y);
extern "C" __device__ bool fp_nonzero(const fp_t &x);
extern "C" __device__ bool fp_iszero(const fp_t &x);
extern "C" __device__ bool fp_isone(const fp_t &x);

#ifdef DEBUG
extern "C" __device__ void fp_print(const char *s, const fp_t &x);
#endif

//Test for integration
void fp();
